"""Shared training plumbing: accelerator, optimizer, schedule, checkpoint/resume, throughput."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from . import faults


def make_accelerator(cfg: dict):
    from accelerate import Accelerator

    t = cfg["train"]
    mp = t.get("mixed_precision", "bf16" if torch.cuda.is_available() else "no")
    if mp in (False, None):  # YAML reads a bare `no` as False
        mp = "no"
    if mp == "bf16" and not torch.cuda.is_available():
        mp = "no"
    if mp == "bf16" and torch.cuda.get_device_capability()[0] < 8:
        mp = "fp16"  # T4/P100: no native bf16; accelerate adds a GradScaler for fp16
        print("[train] GPU has no native bf16; using fp16 mixed precision with loss scaling")
    return Accelerator(mixed_precision=mp, gradient_accumulation_steps=t.get("grad_accum", 1))


def make_optimizer(params, cfg: dict, lr_scale: float = 1.0):
    t = cfg["train"]
    decay, no_decay = [], []
    for n, p in params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n.lower() or n.endswith(".bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": t.get("weight_decay", 0.0)},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=t["lr"] * lr_scale, betas=tuple(t.get("betas", (0.9, 0.95))),
                             eps=t.get("eps", 1e-8))


def warmup_cosine(optimizer, warmup: int, total: int, min_ratio: float = 0.1):
    """Linear warmup then cosine decay to min_ratio * peak."""
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# --------------------------------------------------------------------------- checkpoint / resume

def save_state(accelerator, out_dir: Path, step: int) -> None:
    """Full training state: model, optimizer, scheduler, RNG (sharded under FSDP)."""
    ck = out_dir / "state"
    accelerator.save_state(str(ck))
    if accelerator.is_main_process:
        with open(ck / "trainer.json", "w") as f:
            json.dump({"step": step, "time": time.time()}, f)


def maybe_resume(accelerator, out_dir: Path, optimizer, scheduler) -> int:
    ck = out_dir / "state"
    if not (ck / "trainer.json").exists():
        return 0
    accelerator.load_state(str(ck))
    step = json.loads((ck / "trainer.json").read_text())["step"]
    if faults.on("resume_no_optim"):
        # The bug: weights restored, but Adam moments and the LR schedule start over.
        optimizer.state.clear()
        scheduler.last_epoch = -1
        scheduler.step()
    accelerator.print(f"[resume] restored state at step {step}")
    return step


# --------------------------------------------------------------------------- throughput

class Throughput:
    """Tokens/sec, peak memory, and MFU estimate (6 * params * tokens / time / peak FLOPs)."""

    def __init__(self, n_params: int, peak_tflops: float | None, world: int):
        self.n_params, self.peak, self.world = n_params, peak_tflops, world
        self.tokens, self.t0, self.history = 0, time.time(), []

    def add(self, tokens: int) -> None:
        self.tokens += tokens

    def read(self) -> dict:
        dt = max(1e-6, time.time() - self.t0)
        tps = self.tokens * self.world / dt
        out = {"tok_per_s": tps}
        if self.peak:
            out["mfu"] = 6 * self.n_params * tps / (self.peak * 1e12 * self.world)
        if torch.cuda.is_available():
            out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        self.history.append(tps)
        self.tokens, self.t0 = 0, time.time()
        return out
