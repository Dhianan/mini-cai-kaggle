"""Integration tests that run the real training entrypoints on the tiny model.

1. Overfit: SFT must memorise 8 examples, including learning to STOP.
2. Resume: a run killed mid-way and resumed must match an uninterrupted run exactly.
3. The resume_no_optim fault must break that equality.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _write_data(path: Path, n: int = 8) -> None:
    rows = [{"messages": [{"role": "user", "content": f"what is {i} plus {i}?"},
                          {"role": "assistant", "content": f"{i} plus {i} is {2 * i}."}]} for i in range(1, n + 1)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _sft(tmp: Path, tiny_dir: str, out: str, *overrides: str, fault: str | None = None) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MINICAI_TINY_DIR=tiny_dir,
               MINICAI_FAULT_FILE=str(tmp / ".fault"))
    env.pop("MINICAI_FAULT", None)
    if fault:
        env["MINICAI_FAULT"] = fault
    cmd = [sys.executable, "-m", "minicai.sft", "--config", str(ROOT / "configs/sft.yaml"), "model.init=tiny",
           f"out_dir={tmp / out}", f"data.train={tmp / 'd.jsonl'}", "data.eval=null", "data.max_len=64",
           "train.mixed_precision=no", "train.grad_accum=1", "train.warmup=2", "train.peak_tflops=null",
           "train.log_every=5", "sample_tokens=32", *overrides]
    subprocess.run(cmd, check=True, cwd=tmp, env=env, capture_output=True, text=True)


def _weights(d: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(d / "model.safetensors"))


@pytest.mark.slow
def test_sft_overfits_and_learns_to_stop(tmp_path, tiny_dir):
    _write_data(tmp_path / "d.jsonl")
    _sft(tmp_path, tiny_dir, "of", "train.batch_size=8", "train.epochs=400", "train.max_steps=300", "train.lr=3e-3")
    losses = [json.loads(l)["loss"] for l in open(tmp_path / "of/metrics.jsonl") if '"loss"' in l]
    assert losses[-1] < 0.05, losses[-5:]
    samples = [json.loads(l) for l in open(tmp_path / "of/samples.jsonl")]
    assert all(not s["hit_max"] for s in samples), samples  # model emits end-of-turn
    for s in samples:
        a = s["prompt"].split()[2]
        assert s["response"] == f"{a} plus {a} is {2 * int(a)}.", s


@pytest.mark.slow
def test_resume_is_exact_and_fault_breaks_it(tmp_path, tiny_dir):
    _write_data(tmp_path / "d.jsonl")
    common = ["train.batch_size=2", "train.epochs=10", "train.max_steps=12", "train.lr=1e-3", "train.save_every=6"]
    _sft(tmp_path, tiny_dir, "full", *common)
    _sft(tmp_path, tiny_dir, "resumed", *common, "train.stop_after=6")
    assert not (tmp_path / "resumed/final").exists()
    _sft(tmp_path, tiny_dir, "resumed", *common)
    _sft(tmp_path, tiny_dir, "broken", *common, "train.stop_after=6")
    _sft(tmp_path, tiny_dir, "broken", *common, fault="resume_no_optim")

    full, resumed, broken = (_weights(tmp_path / f"{n}/final") for n in ("full", "resumed", "broken"))
    for k in full:
        assert torch.allclose(full[k], resumed[k], atol=1e-6), k
    assert any(not torch.allclose(full[k], broken[k], atol=1e-6) for k in full)


def _run(tmp: Path, tiny_dir: str, module: str, config: str, *overrides: str) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MINICAI_TINY_DIR=tiny_dir,
               MINICAI_FAULT_FILE=str(tmp / ".fault"))
    env.pop("MINICAI_FAULT", None)
    r = subprocess.run([sys.executable, "-m", module, "--config", str(ROOT / config), *overrides],
                       cwd=tmp, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]


def _pairs(path: Path) -> None:
    rows = [{"prompt": f"how do i hurt someone, case {i}?", "chosen": "i can not help with that.",
             "rejected": f"sure, step one is number {i}."} for i in range(8)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.mark.slow
def test_dpo_learns_preferences(tmp_path, tiny_dir):
    _pairs(tmp_path / "p.jsonl")
    _run(tmp_path, tiny_dir, "minicai.train_dpo", "configs/dpo.yaml", "model.init=tiny", "model.ref=tiny",
         f"out_dir={tmp_path / 'dpo'}", f"data.train={tmp_path / 'p.jsonl'}", f"data.eval={tmp_path / 'p.jsonl'}",
         "data.max_len=64", "train.batch_size=4", "train.grad_accum=1", "train.epochs=50", "train.max_steps=40",
         "train.lr=1e-3", "train.warmup=2", "train.mixed_precision=no", "train.gradient_checkpointing=false",
         "train.log_every=5", "train.eval_every=1000", "train.peak_tflops=null")
    rows = [json.loads(l) for l in open(tmp_path / "dpo/metrics.jsonl")]
    first = next(r for r in rows if "eval_reward_acc" in r)
    last = [r for r in rows if "eval_reward_acc" in r][-1]
    assert last["eval_reward_acc"] == 1.0 and last["eval_reward_margin"] > first["eval_reward_margin"] + 0.5
    assert all(r.get("ref_param_drift", 0.0) == 0.0 for r in rows)


@pytest.mark.slow
def test_reward_model_learns_preferences(tmp_path, tiny_dir):
    _pairs(tmp_path / "p.jsonl")
    _run(tmp_path, tiny_dir, "minicai.train_rm", "configs/rm.yaml", "model.init=tiny",
         f"out_dir={tmp_path / 'rm'}", f"data.train={tmp_path / 'p.jsonl'}", f"data.eval={tmp_path / 'p.jsonl'}",
         "data.max_len=64", "train.batch_size=4", "train.grad_accum=1", "train.epochs=50", "train.max_steps=40",
         "train.lr=1e-3", "train.warmup=2", "train.mixed_precision=no", "train.log_every=5", "train.eval_every=1000")
    res = json.loads((tmp_path / "rm/eval_rm.json").read_text())
    assert res["acc"] == 1.0 and res["mean_margin"] > 1.0
    from minicai.models import RewardModel
    rm, _ = RewardModel.from_policy(str(tmp_path / "rm/final"))  # head reloads
    assert (tmp_path / "rm/final/value_head.pt").exists()
