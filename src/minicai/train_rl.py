"""Day 4: online RL against the reward model with GRPO (group-relative PPO), hand-written.

Each step: sample P prompts x G completions from the current policy, score them
with the RM, normalise rewards within each prompt's group (no value network),
then take clipped policy-gradient steps with a k3 KL penalty to the reference.

    python -m minicai.train_rl --config configs/rl.yaml

Single GPU by design (sampling + 3 models). Watch these together, never alone:
  reward     should rise slowly
  kl         should rise slowly; a spike = instability, stop early (kl_stop)
  len        if reward rises only because len rises -> reward hacking
  entropy    a collapse toward 0 = mode collapse
  judge win rate (run evals) is the ground truth; RM reward is a proxy
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from . import chat, checks, faults
from .config import RunLogger, cli, config_hash, git_hash, set_seed
from .data import collate, read_jsonl, write_jsonl
from .losses import clipped_pg_loss, group_advantages, kl_k3, masked_mean, token_entropy, token_logprobs
from .models import RewardModel, freeze, generate, half_dtype, load_policy, save_checkpoint
from .train_utils import make_optimizer


def _autocast(device: str):
    return torch.autocast("cuda", dtype=half_dtype()) if device == "cuda" else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def rm_scores(rm, tok, prompts: list[str], texts: list[str], bs: int, max_len: int, device: str) -> list[float]:
    out = []
    for i in range(0, len(texts), bs):
        enc = [chat.encode_prompt_response(tok, chat.as_messages(p), t, max_len)
               for p, t in zip(prompts[i:i + bs], texts[i:i + bs])]
        b = collate(enc, tok.pad_token_id)
        with _autocast(device):
            out += rm(b["input_ids"].to(device), b["attention_mask"].to(device)).tolist()
    return out


def build_rollout_batch(tok, prompts: list[str], gens: list[dict]) -> list[dict]:
    """Reuse the exact sampled token ids (no re-tokenisation), and append the
    end-of-turn token when the model stopped on its own so stopping is reinforced."""
    end_id = tok.convert_tokens_to_ids(chat.IM_END)
    rows = []
    for p, g in zip(prompts, gens):
        p_ids = tok(chat.prompt_text(chat.as_messages(p)), add_special_tokens=False)["input_ids"]
        c_ids = g["token_ids"] + ([] if g["hit_max"] else [end_id])
        if not c_ids:
            c_ids = [end_id]
        rows.append({"input_ids": p_ids + c_ids, "labels": [chat.IGNORE] * len(p_ids) + c_ids})
    return rows


def forward_logps(model, batch: dict, device: str, with_entropy: bool = False):
    with _autocast(device):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    lp, mask = token_logprobs(logits, batch["labels"])
    ent = token_entropy(logits, batch["labels"]) if with_entropy else None
    return lp, mask, ent


def main() -> None:
    cfg = cli("GRPO against a learned reward model")
    chat.set_stage("rl")
    t = cfg["train"]
    out_dir = Path(cfg["out_dir"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(t.get("seed", 0))
    rng = random.Random(t.get("seed", 0))
    log = RunLogger(out_dir, cfg, True)

    policy, tok = load_policy(cfg["model"]["init"], dtype="fp32", device=device)
    ref, _ = load_policy(cfg["model"].get("ref", cfg["model"]["init"]), dtype="auto", device=device)
    freeze(ref)
    rm, _ = RewardModel.from_policy(cfg["reward_model"], dtype="auto")
    rm = freeze(rm.to(device))
    if cfg.get("guardrails", True):
        checks.check_template(cfg["model"]["init"], tok)
    if t.get("gradient_checkpointing"):
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False

    prompts = [r["prompt"] for f in cfg["data"]["prompts"] for r in read_jsonl(f, cfg["data"].get("limit_each"))]
    P, G = t["prompts_per_step"], t["group_size"]
    lr_scale = 10.0 if faults.on("rl_lr_10x") else 1.0
    opt = make_optimizer(policy.named_parameters(), cfg, lr_scale)
    # fp16 autocast (T4) needs loss scaling; with bf16 or on CPU the scaler is a no-op passthrough.
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and half_dtype() == torch.float16)
    g = cfg.get("gen", {})
    mb = t.get("micro_batch", 8)
    samples_path = out_dir / "samples.jsonl"
    print(f"[rl] {len(prompts)} prompts, P={P} G={G}, steps={t['steps']}, kl_coef={t['kl_coef']}")

    for step in range(1, t["steps"] + 1):
        batch_prompts = rng.sample(prompts, P)
        gens = generate(policy, tok, [chat.as_messages(p) for p in batch_prompts],
                        max_new_tokens=g.get("max_new_tokens", 256), temperature=g.get("temperature", 1.0),
                        top_p=g.get("top_p", 1.0), batch_size=g.get("batch_size", 16), num_return_sequences=G)
        flat_prompts = [batch_prompts[x["prompt_idx"]] for x in gens]
        texts = [x["text"] for x in gens]
        raw = rm_scores(rm, tok, flat_prompts, texts, mb, cfg.get("max_len", 1024), device)
        lens = [x["n_tokens"] for x in gens]
        shaped = [r - t.get("length_penalty", 0.0) * n - (t.get("truncation_penalty", 0.0) if x["hit_max"] else 0.0)
                  for r, n, x in zip(raw, lens, gens)]
        adv = group_advantages(torch.tensor(shaped).view(P, G), scale=t.get("scale_adv", True)).flatten()

        rows = build_rollout_batch(tok, flat_prompts, gens)
        # Old (behaviour) and reference log-probs, computed once per rollout.
        old_lp, ref_lp = [], []
        policy.eval()
        with torch.no_grad():
            for i in range(0, len(rows), mb):
                b = {k: v.to(device) for k, v in collate(rows[i:i + mb], tok.pad_token_id).items()}
                old_lp.append(forward_logps(policy, b, device)[0])
                ref_lp.append(forward_logps(ref, b, device)[0])
        policy.train()

        stats = {"loss": 0.0, "pg": 0.0, "kl": 0.0, "clip_frac": 0.0, "entropy": 0.0}
        n_upd = 0
        for _ in range(t.get("ppo_epochs", 1)):
            order = list(range(0, len(rows), mb))
            rng.shuffle(order)
            for i in order:
                k = i // mb
                b = {kk: v.to(device) for kk, v in collate(rows[i:i + mb], tok.pad_token_id).items()}
                lp, mask, ent = forward_logps(policy, b, device, with_entropy=True)
                pg, m = clipped_pg_loss(lp, old_lp[k], adv[i:i + mb].to(device), mask, t.get("clip", 0.2))
                kl = masked_mean(kl_k3(lp, ref_lp[k]), mask)
                loss = pg + t["kl_coef"] * kl
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), t.get("grad_clip", 1.0))
                scaler.step(opt)
                scaler.update()
                for key, val in (("loss", loss), ("pg", pg), ("kl", kl), ("clip_frac", m["clip_frac"]), ("entropy", ent)):
                    stats[key] += float(val)
                n_upd += 1
        stats = {k: v / max(1, n_upd) for k, v in stats.items()}
        refusal = sum(checks.is_refusal(x) for x in texts) / len(texts)
        log.log(step, reward=sum(shaped) / len(shaped), rm_raw=sum(raw) / len(raw), len=sum(lens) / len(lens),
                hit_max_frac=sum(x["hit_max"] for x in gens) / len(gens), refusal_rate=refusal,
                reward_len_corr=checks.pearson(raw, [float(n) for n in lens]),
                lr=opt.param_groups[0]["lr"], **stats)

        if step % t.get("sample_every", 10) == 0 or step == 1:
            with open(samples_path, "a") as f:
                for j in range(min(2, P)):
                    f.write(json.dumps({"step": step, "prompt": batch_prompts[j], "response": texts[j * G],
                                        "rm": raw[j * G], "len": lens[j * G]}) + "\n")
        if stats["kl"] > t.get("kl_stop", 1e9):
            log.log(step, event="kl_stop", kl=stats["kl"])
            print(f"[rl] KL {stats['kl']:.3f} > kl_stop {t['kl_stop']}; stopping to protect the policy.")
            break
        if stats["entropy"] < t.get("entropy_floor", 0.0):
            log.log(step, event="entropy_alert", entropy=stats["entropy"])
        if step % t.get("save_every", 100) == 0:
            save_checkpoint(policy, tok, out_dir / f"step_{step}")

    meta = {"stage": "rl", "parent": cfg["model"]["init"], "reward_model": cfg["reward_model"], "git": git_hash(),
            "config_hash": config_hash({k: v for k, v in cfg.items() if not k.startswith("_")}),
            "template_hash": chat.template_hash(tok), "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}
    save_checkpoint(policy, tok, out_dir / "final", meta=meta)
    print(f"[rl] saved {out_dir / 'final'}")
    log.close()


if __name__ == "__main__":
    main()
