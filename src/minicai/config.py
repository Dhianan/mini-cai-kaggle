"""Config loading, run directories, seeding, and metric logging.

Every stage is driven by one YAML file plus optional `key.sub=value` overrides
on the command line. The resolved config, its hash, the git commit, and the
chat-template hash are written next to every checkpoint (meta.json) so any
result can be traced back to exactly what produced it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- config

def _coerce(value: str) -> Any:
    """Parse a CLI override value with YAML rules (ints, floats, bools, lists)."""
    v = yaml.safe_load(value)
    if isinstance(v, str):  # YAML 1.1 reads "1e-3" as a string
        try:
            return float(v)
        except ValueError:
            pass
    return v


def _set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Load a YAML config. Supports `inherit: other.yaml` (relative path) and overrides."""
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must look like key.sub=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, _coerce(v))
    return cfg


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def cli(description: str) -> dict:
    """Standard CLI for every stage: --config path [overrides...]."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg["_config_path"] = args.config
    cfg["_argv"] = sys.argv
    return cfg


# --------------------------------------------------------------------------- repro

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "nogit"


def write_meta(out_dir: str | os.PathLike, cfg: dict, extra: dict | None = None) -> dict:
    meta = {
        "stage": cfg.get("stage"),
        "config_hash": config_hash({k: v for k, v in cfg.items() if not k.startswith("_")}),
        "git": git_hash(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }
    meta.update(extra or {})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta


def read_meta(model_dir: str | os.PathLike) -> dict:
    p = Path(model_dir) / "meta.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# --------------------------------------------------------------------------- logging

class RunLogger:
    """Append-only JSONL metrics + optional Weights & Biases mirror.

    JSONL is the source of truth: it survives crashes, diffs cleanly, and is
    what the incident drills ask you to read.
    """

    def __init__(self, out_dir: str | os.PathLike, cfg: dict, is_main: bool = True):
        self.is_main = is_main
        self.path = Path(out_dir) / "metrics.jsonl"
        self.wandb = None
        if not is_main:
            return
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if cfg.get("wandb", {}).get("enabled"):
            try:
                import wandb

                self.wandb = wandb.init(
                    project=cfg["wandb"].get("project", "mini-cai"),
                    name=cfg.get("run_name"),
                    config={k: v for k, v in cfg.items() if not k.startswith("_")},
                )
            except Exception as e:  # never let logging kill a run
                print(f"[logger] wandb disabled: {e}")

    def log(self, step: int, **metrics: Any) -> None:
        if not self.is_main:
            return
        row = {"step": step, "t": round(time.time(), 2)}
        row.update({k: (float(v) if hasattr(v, "item") else v) for k, v in metrics.items()})
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        if self.wandb is not None:
            self.wandb.log(row, step=step)
        shown = " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()
            if k != "t"
        )
        print(shown, flush=True)

    def close(self) -> None:
        if self.wandb is not None:
            self.wandb.finish()
