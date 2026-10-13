"""Library baselines to check your hand-written loops against (Day 1 and Day 4).

    pip install trl
    python -m minicai.trl_reference sft --config configs/sft.yaml train.max_steps=200 out_dir=runs/sft_trl
    python -m minicai.trl_reference dpo --config configs/dpo.yaml train.max_steps=100 out_dir=runs/dpo_trl

Then compare runs/sft/metrics.jsonl with runs/sft_trl (TRL logs to its own
trainer_state.json). Same data, LR, batch and seed should give loss curves that
agree within noise. If they don't, one of the two has a bug: find it.

TRL's API moves fast; if a keyword below is rejected, check the installed
version's SFTConfig / DPOConfig docs and adjust.
"""
from __future__ import annotations

import sys

from .config import load_config
from .data import read_jsonl
import torch

from .models import load_policy, native_bf16


def _dump_log(trainer, out_dir: str) -> None:
    """TRL keeps its loss history in memory; write it next to ours (metrics.jsonl, optimizer steps)."""
    import json
    from pathlib import Path
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "metrics.jsonl", "w") as f:
        for row in trainer.state.log_history:
            f.write(json.dumps(row) + "\n")


def _sft(cfg: dict) -> None:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rows = []
    for r in read_jsonl(cfg["data"]["train"], cfg["data"].get("limit")):
        m = r["messages"]
        if m and m[-1]["role"] == "assistant":
            rows.append({"prompt": m[:-1], "completion": [m[-1]]})  # loss on completion only
    model, tok = load_policy(cfg["model"]["init"])
    t = cfg["train"]
    half = t.get("mixed_precision") == "bf16" and torch.cuda.is_available()  # same fp16 fallback as our loop on T4
    args = SFTConfig(output_dir=cfg["out_dir"], learning_rate=t["lr"], per_device_train_batch_size=t["batch_size"],
                     gradient_accumulation_steps=t.get("grad_accum", 1), max_steps=t.get("max_steps") or -1,
                     num_train_epochs=t.get("epochs", 1), warmup_steps=t.get("warmup", 0),
                     lr_scheduler_type="cosine", logging_steps=t.get("log_every", 10), seed=t.get("seed", 0),
                     bf16=half and native_bf16(), fp16=half and not native_bf16(),
                     gradient_checkpointing=bool(t.get("gradient_checkpointing")), report_to=[])
    tr = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok)
    tr.train()
    _dump_log(tr, cfg["out_dir"])


def _dpo(cfg: dict) -> None:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    rows = [{"prompt": [{"role": "user", "content": r["prompt"]}],
             "chosen": [{"role": "assistant", "content": r["chosen"]}],
             "rejected": [{"role": "assistant", "content": r["rejected"]}]}
            for r in read_jsonl(cfg["data"]["train"], cfg["data"].get("limit"))]
    model, tok = load_policy(cfg["model"]["init"])
    t = cfg["train"]
    args = DPOConfig(output_dir=cfg["out_dir"], beta=cfg.get("beta", 0.1), learning_rate=t["lr"],
                     per_device_train_batch_size=t["batch_size"], max_steps=t.get("max_steps") or -1,
                     logging_steps=t.get("log_every", 10), seed=t.get("seed", 0), report_to=[])
    tr = DPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok)
    tr.train()
    _dump_log(tr, cfg["out_dir"])


def main() -> None:
    which = sys.argv[1]
    i = sys.argv.index("--config")
    cfg = load_config(sys.argv[i + 1], sys.argv[i + 2:])
    {"sft": _sft, "dpo": _dpo}[which](cfg)


if __name__ == "__main__":
    main()
