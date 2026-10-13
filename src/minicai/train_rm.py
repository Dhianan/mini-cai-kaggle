"""Day 3: train a Bradley-Terry reward model on the AI-labelled preferences.

    python -m minicai.train_rm --config configs/rm.yaml

Eval reports what matters for RL later:
  acc                       held-out pairwise accuracy
  acc_chosen_longer/shorter accuracy split by whether the chosen answer is longer;
                            a big gap means the RM learned "longer is better"
  reward_length_corr        Pearson r between reward and response length
  ece                       calibration of sigmoid(margin) as a probability
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import chat, checks
from .config import RunLogger, cli, config_hash, git_hash, set_seed
from .data import batches, collate, load_pairs
from .losses import bradley_terry_loss
from .models import RewardModel
from .train_utils import make_accelerator, make_optimizer, warmup_cosine


def encode_pairs(tok, rows: list[dict], max_len: int) -> list[dict]:
    out = []
    for r in rows:
        msgs = chat.as_messages(r["prompt"])
        c = chat.encode_prompt_response(tok, msgs, r["chosen"], max_len)
        j = chat.encode_prompt_response(tok, msgs, r["rejected"], max_len)
        out.append({"chosen": c, "rejected": j, "len_c": len(r["chosen"]), "len_r": len(r["rejected"])})
    return out


def pair_batch(items: list[dict], pad_id: int, device) -> dict:
    b = collate([i["chosen"] for i in items] + [i["rejected"] for i in items], pad_id)
    return {k: v.to(device) for k, v in b.items()}


@torch.no_grad()
def evaluate(rm, items: list[dict], pad_id: int, bs: int, device) -> dict:
    rm.eval()
    margins, rewards, lengths, longer = [], [], [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        b = pair_batch(chunk, pad_id, device)
        r = rm(b["input_ids"], b["attention_mask"])
        rc, rr = r[: len(chunk)], r[len(chunk):]
        margins += (rc - rr).tolist()
        rewards += rc.tolist() + rr.tolist()
        lengths += [x["len_c"] for x in chunk] + [x["len_r"] for x in chunk]
        longer += [x["len_c"] > x["len_r"] for x in chunk]
    rm.train()
    if not margins:
        return {}
    correct = [m > 0 for m in margins]
    def acc_where(flag):
        sel = [c for c, l in zip(correct, longer) if l == flag]
        return sum(sel) / len(sel) if sel else float("nan")
    # Expected calibration error over 5 bins of predicted P(chosen wins).
    probs = [torch.sigmoid(torch.tensor(m)).item() for m in margins]
    ece, bins = 0.0, 5
    for k in range(bins):
        lo, hi = 0.5 + k * 0.5 / bins, 0.5 + (k + 1) * 0.5 / bins
        idx = [i for i, p in enumerate(probs) if lo <= max(p, 1 - p) < hi or (k == bins - 1 and max(p, 1 - p) == 1)]
        if idx:
            conf = sum(max(probs[i], 1 - probs[i]) for i in idx) / len(idx)
            accb = sum(correct[i] for i in idx) / len(idx)
            ece += len(idx) / len(probs) * abs(conf - accb)
    return {"acc": sum(correct) / len(correct), "acc_chosen_longer": acc_where(True),
            "acc_chosen_shorter": acc_where(False), "reward_length_corr": checks.pearson(rewards, lengths),
            "ece": ece, "mean_margin": sum(margins) / len(margins), "n": len(margins)}


def main() -> None:
    cfg = cli("Bradley-Terry reward model")
    chat.set_stage("rm")
    t, d = cfg["train"], cfg["data"]
    out_dir = Path(cfg["out_dir"])
    acc = make_accelerator(cfg)
    set_seed(t.get("seed", 0))
    log = RunLogger(out_dir, cfg, acc.is_main_process)

    rm, tok = RewardModel.from_policy(cfg["model"]["init"], dtype=cfg["model"].get("dtype", "fp32"))
    if cfg.get("guardrails", True):
        checks.check_template(cfg["model"]["init"], tok)
    if t.get("gradient_checkpointing"):
        rm.backbone.gradient_checkpointing_enable()
    train = encode_pairs(tok, load_pairs(d["train"], "rm", d.get("limit")), d.get("max_len", 1024))
    evals = encode_pairs(tok, load_pairs(d["eval"], "rm_eval"), d.get("max_len", 1024)) if d.get("eval") else []
    bs, world, rank = t["batch_size"], acc.num_processes, acc.process_index
    micro_per_epoch = max(1, len(train) // (bs * world))
    total = min(t.get("max_steps") or 10**12, t.get("epochs", 1) * micro_per_epoch)
    opt = make_optimizer(rm.named_parameters(), cfg)
    sched = warmup_cosine(opt, t.get("warmup", 0), max(1, total // t.get("grad_accum", 1)))
    rm, opt = acc.prepare(rm, opt)
    acc.print(f"[rm] {len(train)} train pairs, {len(evals)} eval pairs, {total} steps")
    log.log(0, **{f"eval_{k}": v for k, v in evaluate(acc.unwrap_model(rm), evals, tok.pad_token_id, bs, acc.device).items()})
    rm.train()
    for step, gb in batches(train, bs * world, True, t.get("seed", 0), t.get("epochs", 1)):
        if step >= total:
            break
        local = gb[rank::world]
        b = pair_batch(local, tok.pad_token_id, acc.device)
        with acc.accumulate(rm):
            with acc.autocast():
                r = rm(b["input_ids"], b["attention_mask"])
                loss, m = bradley_terry_loss(r[: len(local)], r[len(local):])
                # Small penalty keeps rewards centred near 0 (stabilises RL later).
                loss = loss + t.get("center_coef", 0.0) * (r ** 2).mean()
            acc.backward(loss)
            if acc.sync_gradients:
                acc.clip_grad_norm_(rm.parameters(), t.get("grad_clip", 1.0))
            opt.step()
            if acc.sync_gradients:
                sched.step()
            opt.zero_grad(set_to_none=True)
        if (step + 1) % t.get("log_every", 10) == 0:
            log.log(step + 1, loss=loss.item(), lr=sched.get_last_lr()[0], **{k: v.item() for k, v in m.items()})
        if evals and (step + 1) % t.get("eval_every", 100) == 0:
            log.log(step + 1, **{f"eval_{k}": v for k, v in evaluate(acc.unwrap_model(rm), evals, tok.pad_token_id, bs, acc.device).items()})

    final = evaluate(acc.unwrap_model(rm), evals, tok.pad_token_id, bs, acc.device)
    log.log(total, **{f"eval_{k}": v for k, v in final.items()})
    if acc.is_main_process:
        meta = {"stage": "rm", "parent": cfg["model"]["init"], "git": git_hash(),
                "config_hash": config_hash({k: v for k, v in cfg.items() if not k.startswith("_")}),
                "template_hash": chat.template_hash(tok), "eval": final,
                "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}
        acc.unwrap_model(rm).save(out_dir / "final", tok, meta)
        (out_dir / "eval_rm.json").write_text(json.dumps(final, indent=2))
        print("[rm]", json.dumps(final, indent=2))
        if final and abs(final.get("reward_length_corr", 0)) > 0.5:
            print("[rm] WARNING: |reward-length corr| > 0.5. RL will likely learn to pad answers.")
    log.close()


if __name__ == "__main__":
    main()
