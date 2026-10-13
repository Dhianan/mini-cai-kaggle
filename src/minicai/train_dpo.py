"""Day 4: Direct Preference Optimization, hand-written.

    python -m minicai.train_dpo --config configs/dpo.yaml
    for b in 0.05 0.1 0.5; do python -m minicai.train_dpo --config configs/dpo.yaml beta=$b out_dir=runs/dpo_b$b; done
    accelerate launch --config_file configs/accelerate/fsdp.yaml -m minicai.train_dpo --config configs/dpo.yaml

What to watch in metrics.jsonl:
  reward_acc        should rise well above 0.5 on train AND eval
  reward_margin     grows; if train grows but eval stalls -> overfitting
  logp_chosen       often FALLS too (both go down, rejected faster); a steep fall
                    in logp_chosen = likelihood displacement, outputs degrade
  ref_param_drift   must be exactly 0.0 (frozen reference)
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch

from . import chat, checks, faults
from .config import RunLogger, cli, config_hash, git_hash, set_seed
from .data import batches, collate, load_pairs
from .losses import dpo_loss, sequence_logprobs
from .models import assert_frozen, freeze, load_policy, param_norm, save_checkpoint
from .train_rm import encode_pairs
from .train_utils import Throughput, make_accelerator, make_optimizer, maybe_resume, save_state, warmup_cosine


def logps_for(model, items: list[dict], pad_id: int, device, average: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward over [chosen; rejected] -> (logp_chosen [B], logp_rejected [B])."""
    b = collate([i["chosen"] for i in items] + [i["rejected"] for i in items], pad_id)
    b = {k: v.to(device) for k, v in b.items()}
    logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
    lp = sequence_logprobs(logits, b["labels"], average=average)
    return lp[: len(items)], lp[len(items):]


def main() -> None:
    cfg = cli("Direct Preference Optimization")
    chat.set_stage("dpo")
    t, d = cfg["train"], cfg["data"]
    out_dir = Path(cfg["out_dir"])
    acc = make_accelerator(cfg)
    set_seed(t.get("seed", 0))
    log = RunLogger(out_dir, cfg, acc.is_main_process)
    beta, ls, avg = cfg.get("beta", 0.1), cfg.get("label_smoothing", 0.0), cfg.get("length_normalize", False)

    policy, tok = load_policy(cfg["model"]["init"], dtype=cfg["model"].get("dtype", "fp32"))
    ref, _ = load_policy(cfg["model"].get("ref", cfg["model"]["init"]), dtype=cfg["model"].get("ref_dtype", "auto"))
    unfrozen = faults.on("ref_unfrozen")
    if not unfrozen:
        freeze(ref)
    if cfg.get("guardrails", True):
        checks.check_template(cfg["model"]["init"], tok)
        assert_frozen(ref)
    if t.get("gradient_checkpointing"):
        policy.gradient_checkpointing_enable()
        policy.config.use_cache = False

    train = encode_pairs(tok, load_pairs(d["train"], "dpo", d.get("limit")), d.get("max_len", 1024))
    evals = encode_pairs(tok, load_pairs(d["eval"], "dpo_eval"), d.get("max_len", 1024)) if d.get("eval") else []
    bs, world, rank = t["batch_size"], acc.num_processes, acc.process_index
    total = min(t.get("max_steps") or 10**12, t.get("epochs", 1) * max(1, len(train) // (bs * world)))

    params = list(policy.named_parameters()) + (list(ref.named_parameters()) if unfrozen else [])
    opt = make_optimizer(params, cfg)
    sched = warmup_cosine(opt, t.get("warmup", 0), max(1, total // t.get("grad_accum", 1)))
    policy, opt = acc.prepare(policy, opt)
    ref = acc.prepare(ref) if "FSDP" in str(acc.distributed_type) else ref.to(acc.device)
    acc.register_for_checkpointing(sched)
    start = maybe_resume(acc, out_dir, opt, sched)
    ref_norm0 = param_norm(ref)
    tp = Throughput(sum(p.numel() for p in acc.unwrap_model(policy).parameters()), t.get("peak_tflops"), world)
    ref_ctx = (lambda: contextlib.nullcontext()) if unfrozen else torch.no_grad

    def run_eval(step: int) -> None:
        if not evals:
            return
        policy.eval()
        sums, n = {}, 0
        with torch.no_grad():
            for i in range(0, len(evals), bs):
                chunk = evals[i:i + bs]
                with acc.autocast():
                    pc, pr = logps_for(policy, chunk, tok.pad_token_id, acc.device, avg)
                    rc, rr = logps_for(ref, chunk, tok.pad_token_id, acc.device, avg)
                _, m = dpo_loss(pc, pr, rc, rr, beta, ls)
                for k, v in m.items():
                    sums[k] = sums.get(k, 0.0) + v.item() * len(chunk)
                n += len(chunk)
        policy.train()
        log.log(step, **{f"eval_{k}": v / n for k, v in sums.items()})

    acc.print(f"[dpo] beta={beta} {len(train)} pairs, {total} steps, ref frozen={not unfrozen}")
    if start == 0:
        run_eval(0)
    policy.train()
    for step, gb in batches(train, bs * world, True, t.get("seed", 0), t.get("epochs", 1), start_step=start):
        if step >= total:
            break
        local = gb[rank::world]
        with acc.accumulate(policy):
            with acc.autocast():
                pc, pr = logps_for(policy, local, tok.pad_token_id, acc.device, avg)
                with ref_ctx():
                    rc, rr = logps_for(ref, local, tok.pad_token_id, acc.device, avg)
                loss, m = dpo_loss(pc, pr, rc, rr, beta, ls)
            acc.backward(loss)
            if acc.sync_gradients:
                acc.clip_grad_norm_(policy.parameters(), t.get("grad_clip", 1.0))
            opt.step()
            if acc.sync_gradients:
                sched.step()
            opt.zero_grad(set_to_none=True)
        tp.add(sum(len(i["chosen"]["input_ids"]) + len(i["rejected"]["input_ids"]) for i in local))
        if (step + 1) % t.get("log_every", 10) == 0:
            log.log(step + 1, loss=loss.item(), lr=sched.get_last_lr()[0],
                    ref_param_drift=abs(param_norm(ref) - ref_norm0), **{k: v.item() for k, v in m.items()},
                    **tp.read())
        if (step + 1) % t.get("eval_every", 100) == 0:
            run_eval(step + 1)
        if (step + 1) % t.get("save_every", 500) == 0:
            save_state(acc, out_dir, step + 1)

    run_eval(total)
    meta = {"stage": "dpo", "parent": cfg["model"]["init"], "beta": beta, "git": git_hash(),
            "config_hash": config_hash({k: v for k, v in cfg.items() if not k.startswith("_")}),
            "template_hash": chat.template_hash(tok), "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}
    save_checkpoint(policy, tok, out_dir / "final", acc, meta)
    acc.print(f"[dpo] saved {out_dir / 'final'}")
    log.close()


if __name__ == "__main__":
    main()
