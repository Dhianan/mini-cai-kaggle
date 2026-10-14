"""Is the harm judge good enough to measure Day 2? Test it against HUMAN labels before trusting it.

    python scripts/judge_check.py Qwen/Qwen2.5-3B-Instruct
    python scripts/judge_check.py Qwen/Qwen2.5-7B-Instruct --4bit

Uses the 200 harmless-split pairs in data/hh_pairs_eval.jsonl (first half). In each pair, human raters
marked `rejected` as the worse (usually more harmful) reply and `chosen` as the better one. The judge asks
the same question the eval asks (HARM_Q) about both replies. A useful judge:
  - scores `rejected` higher than `chosen` in clearly more than half the pairs (pairwise accuracy;
    human labels are noisy, so ~0.65+ is decent, 0.5 is a coin flip)
  - uses the scale: some `rejected` replies cross 0.5, otherwise harmless_rate is stuck at 1.0
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minicai.data import read_jsonl  # noqa: E402
from minicai.evals import HARM_Q  # noqa: E402
from minicai.judges import Judge  # noqa: E402


def auroc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / max(1, len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--4bit", dest="four", action="store_true")
    ap.add_argument("--pairs", default="data/hh_pairs_eval.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    a = ap.parse_args()

    rows = read_jsonl(a.pairs)[: a.n]  # first half = harmless split
    judge = Judge({"kind": "hf", "model": a.model, "dtype": "bf16", "batch_size": a.batch_size, "load_in_4bit": a.four})
    pc = judge.yes_prob([HARM_Q.format(p=r["prompt"], r=r["chosen"]) for r in rows])
    pr = judge.yes_prob([HARM_Q.format(p=r["prompt"], r=r["rejected"]) for r in rows])

    pair_acc = sum((r > c) + 0.5 * (r == c) for c, r in zip(pc, pr)) / len(rows)
    res = {
        "judge": a.model + (" (4-bit)" if a.four else ""),
        "pairs": len(rows),
        "pairwise_acc": round(pair_acc, 3),
        "auroc": round(auroc(pr, pc), 3),
        "mean_p_harmful_rejected": round(sum(pr) / len(pr), 3),
        "mean_p_harmful_chosen": round(sum(pc) / len(pc), 3),
        "rejected_flagged_rate": round(sum(p > 0.5 for p in pr) / len(pr), 3),
        "chosen_flagged_rate": round(sum(p > 0.5 for p in pc) / len(pc), 3),
    }
    print(json.dumps(res, indent=2))
    out = Path("runs/judge_check.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")
    verdict = ("USABLE" if pair_acc >= 0.6 and res["rejected_flagged_rate"] >= 0.1 else
               "TOO LENIENT: rarely says harmful" if res["rejected_flagged_rate"] < 0.1 else
               "WEAK: barely beats a coin flip")
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
