"""Day 6 fault injector.

Each fault is a realistic post-training bug wired into the real code path
behind `faults.on("<name>")`. Arm one blind, run the pipeline, and diagnose it
from metrics.jsonl, eval_results.json, and samples only.

    python -m minicai.faults arm            # picks one at random, hides it
    python -m minicai.faults arm --name eos_dropped
    python -m minicai.faults status         # "armed" / "clean" (never the name)
    python -m minicai.faults reveal         # after you've written your diagnosis
    python -m minicai.faults disarm

The active fault lives in `.fault` at the repo root (or MINICAI_FAULT env var).
"""
from __future__ import annotations

import argparse
import base64
import os
import random
from pathlib import Path

FAULT_FILE = Path(os.environ.get("MINICAI_FAULT_FILE", ".fault"))

FAULTS: dict[str, str] = {
    "prompt_loss": "SFT loss is computed on prompt tokens too (loss mask broken)",
    "template_mismatch": "DPO/RL render prompts with a different chat template than SFT",
    "ref_unfrozen": "DPO reference model is trainable and in the optimizer",
    "eos_dropped": "End-of-turn token is dropped from assistant targets",
    "rl_lr_10x": "RL learning rate silently multiplied by 10",
    "rm_length_bias": "Reward-model training pairs relabelled so the longer response wins",
    "eval_leak": "Eval prompts are leaked into the SFT training set",
    "padding_right": "Batched generation uses right padding",
    "resume_no_optim": "Resume restores weights but resets optimizer and LR schedule",
    "pref_swap_half": "Chosen/rejected swapped for half of the preference pairs",
}


def _encode(name: str) -> str:
    # Obfuscated so `cat .fault` doesn't spoil the drill.
    return base64.b64encode(name[::-1].encode()).decode()


def _decode(blob: str) -> str:
    return base64.b64decode(blob.strip()).decode()[::-1]


def active() -> str | None:
    env = os.environ.get("MINICAI_FAULT")
    if env:
        return env
    if FAULT_FILE.exists():
        try:
            return _decode(FAULT_FILE.read_text())
        except Exception:
            return None
    return None


def on(name: str) -> bool:
    if name not in FAULTS:
        raise KeyError(f"Unknown fault {name!r}")
    return active() == name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["arm", "status", "reveal", "disarm", "list"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.cmd == "list":
        for k, v in FAULTS.items():
            print(f"{k:20s} {v}")
    elif args.cmd == "arm":
        name = args.name or random.SystemRandom().choice(sorted(FAULTS))
        if name not in FAULTS:
            raise SystemExit(f"Unknown fault {name!r}")
        FAULT_FILE.write_text(_encode(name))
        print("Fault armed. Run the pipeline and diagnose from logs. Time-box: 45 min.")
    elif args.cmd == "status":
        print("armed" if active() else "clean")
    elif args.cmd == "reveal":
        name = active()
        print(f"{name}: {FAULTS[name]}" if name else "clean")
    elif args.cmd == "disarm":
        FAULT_FILE.unlink(missing_ok=True)
        print("clean")


if __name__ == "__main__":
    main()
