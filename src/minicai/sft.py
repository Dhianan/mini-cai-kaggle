"""Day 1 (SFT) and Day 2 (SL-CAI re-SFT). Hand-written loop on top of accelerate,
so the same file runs on 1 GPU, DDP, or FSDP:

    python -m minicai.sft --config configs/sft.yaml
    accelerate launch --config_file configs/accelerate/fsdp.yaml -m minicai.sft --config configs/sft.yaml

Kill it at any point and rerun the same command: it resumes from runs/<name>/state.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from . import checks, chat
from .config import RunLogger, cli, config_hash, git_hash, set_seed
from .data import batches, collate, pack, read_jsonl, write_jsonl
from .losses import sft_loss
from .models import generate, load_policy, save_checkpoint
from .train_utils import Throughput, make_accelerator, make_optimizer, maybe_resume, save_state, warmup_cosine


def encode_rows(tok, rows: list[dict], max_len: int) -> list[dict]:
    out = []
    for r in rows:
        e = chat.encode_chat(tok, r["messages"], max_len)
        if any(l != chat.IGNORE for l in e["labels"]):
            out.append(e)
    return out


@torch.no_grad()
def eval_loss(model, examples: list[dict], tok, bs: int, accelerator) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(examples), bs):
        b = collate(examples[i:i + bs], tok.pad_token_id)
        b = {k: v.to(accelerator.device) for k, v in b.items()}
        with accelerator.autocast():
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        total += sft_loss(logits, b["labels"]).item()
        n += 1
    model.train()
    return total / max(1, n)


def main() -> None:
    cfg = cli("Supervised fine-tuning with an assistant-only loss mask")
    chat.set_stage(cfg.get("stage", "sft"))
    t, d = cfg["train"], cfg["data"]
    out_dir = Path(cfg["out_dir"])
    acc = make_accelerator(cfg)
    set_seed(t.get("seed", 0))
    log = RunLogger(out_dir, cfg, acc.is_main_process)

    model, tok = load_policy(cfg["model"]["init"], dtype=cfg["model"].get("dtype", "fp32"))
    if t.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    rows = read_jsonl(d["train"], d.get("limit"))
    examples = encode_rows(tok, rows, d.get("max_len", 1024))
    if cfg.get("guardrails", True):
        acc.print("[guard] loss mask:", checks.check_loss_mask(tok, examples))
    if d.get("pack"):
        examples = pack(examples, d["max_len"], tok.convert_tokens_to_ids(chat.IM_END))
    eval_examples = encode_rows(tok, read_jsonl(d["eval"], d.get("eval_limit", 200)), d.get("max_len", 1024)) if d.get("eval") else []

    world, rank = acc.num_processes, acc.process_index
    bs, accum = t["batch_size"], t.get("grad_accum", 1)
    micro_per_epoch = len(examples) // (bs * world)
    if micro_per_epoch == 0:
        raise SystemExit(f"Not enough examples ({len(examples)}) for batch {bs} x {world} processes.")
    total_micro = min(t.get("max_steps") or 10**12, t.get("epochs", 1) * micro_per_epoch)
    total_opt = max(1, total_micro // accum)

    optimizer = make_optimizer(model.named_parameters(), cfg)
    scheduler = warmup_cosine(optimizer, t.get("warmup", 0), total_opt, t.get("min_lr_ratio", 0.1))
    model, optimizer = acc.prepare(model, optimizer)
    acc.register_for_checkpointing(scheduler)
    start = maybe_resume(acc, out_dir, optimizer, scheduler)

    n_params = sum(p.numel() for p in acc.unwrap_model(model).parameters())
    tp = Throughput(n_params, t.get("peak_tflops"), world)
    acc.print(f"[sft] {len(examples)} examples, {total_micro} micro-steps, {total_opt} optimizer steps, "
              f"{n_params/1e6:.1f}M params, world={world}")
    model.train()
    last_loss = float("nan")
    for step, global_batch in batches(examples, bs * world, shuffle=True, seed=t.get("seed", 0),
                                      epochs=t.get("epochs", 1), start_step=start):
        if step >= total_micro:
            break
        local = global_batch[rank::world]
        b = {k: v.to(acc.device) for k, v in collate(local, tok.pad_token_id).items()}
        with acc.accumulate(model):
            with acc.autocast():
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                loss = sft_loss(logits, b["labels"])
            acc.backward(loss)
            gnorm = None
            if acc.sync_gradients:
                gnorm = acc.clip_grad_norm_(model.parameters(), t.get("grad_clip", 1.0))
            optimizer.step()
            if acc.sync_gradients:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        tp.add(int(b["attention_mask"].sum()))
        last_loss = loss.item()
        if not torch.isfinite(loss):
            log.log(step, event="nan_loss", loss=last_loss)
            raise SystemExit("Loss is not finite; inspect metrics.jsonl around this step.")
        if (step + 1) % t.get("log_every", 10) == 0:
            log.log(step + 1, loss=last_loss, lr=scheduler.get_last_lr()[0],
                    grad_norm=float(gnorm) if gnorm is not None else None, **tp.read())
        if eval_examples and (step + 1) % t.get("eval_every", 200) == 0:
            log.log(step + 1, eval_loss=eval_loss(model, eval_examples, tok, bs, acc))
        if (step + 1) % t.get("save_every", 500) == 0:
            save_state(acc, out_dir, step + 1)
        if t.get("stop_after") and step + 1 >= t["stop_after"]:
            acc.print(f"[sft] stop_after={t['stop_after']} reached (simulated crash); rerun to resume.")
            return

    final = out_dir / "final"
    meta = {"stage": cfg.get("stage", "sft"), "parent": cfg["model"]["init"], "config_hash": config_hash(
        {k: v for k, v in cfg.items() if not k.startswith("_")}), "git": git_hash(),
        "template_hash": chat.template_hash(tok), "n_examples": len(examples),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}
    save_checkpoint(model, tok, final, acc, meta)
    if eval_examples:
        log.log(total_micro, eval_loss=eval_loss(model, eval_examples, tok, bs, acc))

    if acc.is_main_process:
        summary = {"final_train_loss": last_loss, "micro_steps": total_micro,
                   "mean_tok_per_s": sum(tp.history) / max(1, len(tp.history)), "world": world,
                   "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        # A few samples for eyeballing (Day 1 done-criterion: "chats coherently").
        # Skipped under FSDP: params are sharded; run evals on final/ instead.
        if "FSDP" not in str(acc.distributed_type):
            probe = [r["messages"][:1] for r in random.Random(0).sample(rows, min(4, len(rows)))]
            outs = generate(acc.unwrap_model(model), tok, probe, max_new_tokens=cfg.get("sample_tokens", 128),
                            greedy=True)
            write_jsonl(out_dir / "samples.jsonl",
                        [{"prompt": p[0]["content"], "response": o["text"], "hit_max": o["hit_max"]}
                         for p, o in zip(probe, outs)])
        print(f"[sft] saved {final} in {time.strftime('%H:%M:%S')}")
    log.close()


if __name__ == "__main__":
    main()
