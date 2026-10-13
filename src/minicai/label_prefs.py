"""Day 3: RLAIF — sample response pairs from the SL-CAI policy and let an AI
labeler pick the better one against a sampled constitutional principle.

Labeler quality is measured BEFORE labelling anything:
  - sanity accuracy on known-answer pairs (should be ~1.0)
  - agreement with human labels from Anthropic/hh-rlhf
  - position-bias rate (verdict flips when A/B are swapped)
  - length bias (how often the longer response wins)

    python -m minicai.label_prefs --config configs/label_prefs.yaml
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from . import chat
from .cai_revise import load_constitution
from .config import cli, set_seed
from .data import read_jsonl, write_jsonl
from .judges import Judge
from .models import generate, load_policy


def labeler_quality(judge: Judge, pairs: list[dict], principle: str) -> dict:
    if not pairs:
        return {}
    res = judge.prefer([p["prompt"] for p in pairs], [p["chosen"] for p in pairs], [p["rejected"] for p in pairs],
                       [principle] * len(pairs))
    acc = sum(r["p_a"] > 0.5 for r in res) / len(res)
    consistent = sum(r["consistent"] for r in res) / len(res)
    return {"n": len(res), "accuracy": acc, "order_consistency": consistent}


def main() -> None:
    cfg = cli("AI-feedback preference labelling")
    chat.set_stage("label_prefs")
    set_seed(cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, tok = load_policy(cfg["model"]["init"], dtype="auto", device=device)
    judge = Judge(cfg["labeler"], policy, tok)
    constitution = load_constitution(cfg["constitution"])
    pools = {tag: [p for p in constitution if tag in p["tags"]] for tag in ("harmless", "helpful")}
    out = Path(cfg["out"]["dir"])

    # ---- 1. labeler quality
    q = cfg.get("agreement", {})
    lim = q.get("limit", 200)
    report = {
        "sanity": labeler_quality(judge, read_jsonl(q["sanity"], lim) if q.get("sanity") else [],
                                  "Choose the response that actually answers the human's message."),
        "human_agreement": labeler_quality(judge, read_jsonl(q["human_pairs"], lim) if q.get("human_pairs") else [],
                                           "Choose the response that is more helpful, honest, and harmless."),
    }
    print("[labeler]", json.dumps(report))
    min_acc = cfg.get("sanity_min_acc", 0.8)
    if cfg.get("guardrails", True) and report["sanity"] and report["sanity"]["accuracy"] < min_acc:
        raise SystemExit(f"Guardrail: labeler sanity accuracy {report['sanity']['accuracy']:.2f} < {min_acc}. "
                         "Fix the labeler (model, prompt, answer parsing) before generating preferences.")

    # ---- 2. sample pairs
    d, g = cfg["data"], cfg.get("gen", {})
    items = [(r["prompt"], "harmless") for r in read_jsonl(d["harmless_prompts"], d.get("limit_each"))]
    items += [(r["prompt"], "helpful") for r in read_jsonl(d["helpful_prompts"], d.get("limit_each"))]
    rng.shuffle(items)
    convs = [chat.as_messages(p) for p, _ in items]
    outs = generate(policy, tok, convs, max_new_tokens=g.get("max_new_tokens", 256),
                    temperature=g.get("temperature", 1.0), top_p=g.get("top_p", 1.0),
                    batch_size=g.get("batch_size", 16), num_return_sequences=2)
    a_txt, b_txt = [o["text"] for o in outs[0::2]], [o["text"] for o in outs[1::2]]

    # ---- 3. label
    principles = [rng.choice(pools[tag]) for _, tag in items]
    res = judge.prefer([p for p, _ in items], a_txt, b_txt, [c["preference"] for c in principles])
    rows, dropped = [], {"identical": 0, "inconsistent": 0, "low_confidence": 0, "empty": 0}
    for (p, tag), a, b, c, r in zip(items, a_txt, b_txt, principles, res):
        if not a or not b:
            dropped["empty"] += 1
            continue
        if a == b:
            dropped["identical"] += 1
            continue
        if cfg.get("drop_inconsistent", True) and not r["consistent"]:
            dropped["inconsistent"] += 1
            continue
        if abs(r["p_a"] - 0.5) < cfg.get("min_confidence", 0.1):
            dropped["low_confidence"] += 1
            continue
        chosen, rejected = (a, b) if r["p_a"] > 0.5 else (b, a)
        rows.append({"prompt": p, "chosen": chosen, "rejected": rejected, "pool": tag, "principle": c["id"],
                     "p_chosen": max(r["p_a"], 1 - r["p_a"])})

    longer = sum(len(r["chosen"]) > len(r["rejected"]) for r in rows) / max(1, len(rows))
    report.update({"pairs_sampled": len(items), "pairs_kept": len(rows), "dropped": dropped,
                   "position_bias_rate": 1 - sum(r["consistent"] for r in res) / max(1, len(res)),
                   "chosen_is_longer_rate": longer})
    rng.shuffle(rows)
    n_eval = max(1, int(len(rows) * d.get("eval_frac", 0.1))) if len(rows) > 1 else 0
    write_jsonl(out / "prefs_eval.jsonl", rows[:n_eval])
    write_jsonl(out / "prefs_train.jsonl", rows[n_eval:])
    (out / "labeler_report.json").write_text(json.dumps(report, indent=2))
    print("[labeler]", json.dumps(report, indent=2))
    if longer > 0.7:
        print("[labeler] WARNING: longer response wins >70% of the time. Expect a length-biased RM (Day 4 hacking).")


if __name__ == "__main__":
    main()
