"""Day 1 check: does the hand-written SFT loop learn like TRL's?

    python scripts/compare_trl.py runs/sft runs/sft_trl --grad-accum 8

Ours logs one micro-batch loss every `log_every` micro-steps; TRL logs the mean loss per optimizer
step. We convert our steps to optimizer steps and average both into windows of 20 optimizer steps.
Expect the curves to agree within ~0.05-0.1 early on. They drift slightly later because TRL's cosine
schedule spans 200 steps while ours spans the full run, so compare the first ~100 steps most closely.
"""
import argparse
import json
from pathlib import Path


def ours(path: Path, ga: int) -> dict[int, float]:
    out = {}
    for line in open(path / "metrics.jsonl"):
        r = json.loads(line)
        if "loss" in r and "event" not in r:
            out[r["step"] // ga] = r["loss"]
    return out


def trl(path: Path) -> dict[int, float]:
    return {r["step"]: r["loss"] for r in map(json.loads, open(path / "metrics.jsonl")) if "loss" in r}


def windows(d: dict[int, float], w: int, upto: int) -> list[float | None]:
    res = []
    for s in range(0, upto, w):
        v = [x for k, x in d.items() if s < k <= s + w]
        res.append(sum(v) / len(v) if v else None)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ours")
    ap.add_argument("trl")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--upto", type=int, default=200)
    ap.add_argument("--window", type=int, default=20)
    a = ap.parse_args()
    o, t = windows(ours(Path(a.ours), a.grad_accum), a.window, a.upto), windows(trl(Path(a.trl)), a.window, a.upto)
    print(f"{'opt steps':>12} {'ours':>8} {'trl':>8} {'diff':>8}")
    diffs = []
    for i, (x, y) in enumerate(zip(o, t)):
        d = None if x is None or y is None else x - y
        if d is not None and i * a.window < 100:
            diffs.append(abs(d))
        f = lambda v: f"{v:8.3f}" if v is not None else f"{'-':>8}"
        print(f"{i * a.window + 1:>5}-{(i + 1) * a.window:<6} {f(x)} {f(y)} {f(d)}")
    if diffs:
        m = max(diffs)
        print(f"\nmax |diff| in first 100 steps: {m:.3f} -> " +
              ("AGREE within noise" if m < 0.15 else "INVESTIGATE: loss mask, label shift, or LR schedule differs"))


if __name__ == "__main__":
    main()
