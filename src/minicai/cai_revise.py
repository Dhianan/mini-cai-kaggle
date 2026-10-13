"""Day 2: Constitutional AI supervised phase — critique and revise.

For each red-team prompt:
  1. the SFT policy answers
  2. the critic critiques that answer against a randomly sampled principle
  3. the critic rewrites the answer to fix what the critique found
  (repeat 2-3 for `revision_rounds`, a new principle each round)

The final revisions are mixed with helpful SFT data (to stop the model becoming
evasive) and written as a new SFT dataset. Then re-run SFT:

    python -m minicai.cai_revise --config configs/cai_revise.yaml
    python -m minicai.sft --config configs/slcai.yaml
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import torch
import yaml

from . import chat
from .config import cli, set_seed
from .data import read_jsonl, write_jsonl
from .judges import Judge
from .models import generate, load_policy

_PREFIX = re.compile(r"^\s*(revised response|revision|rewritten response|response)\s*:\s*", re.IGNORECASE)


def load_constitution(path: str) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["principles"]


def clean_revision(text: str) -> str:
    return _PREFIX.sub("", text).strip().strip('"').strip()


def main() -> None:
    cfg = cli("CAI critique -> revision data generation")
    chat.set_stage("cai_revise")
    set_seed(cfg.get("seed", 0))
    rng = random.Random(cfg.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, tok = load_policy(cfg["model"]["init"], dtype="auto", device=device)
    critic = Judge(cfg["critic"], policy, tok)
    principles = [p for p in load_constitution(cfg["constitution"]) if set(p["tags"]) & set(cfg.get("tags", ["harmless"]))]
    g = cfg.get("gen", {})

    prompts = [r["prompt"] for r in read_jsonl(cfg["data"]["prompts"], cfg["data"].get("limit"))]
    convs = [chat.as_messages(p) for p in prompts]
    print(f"[cai] {len(prompts)} prompts, {len(principles)} principles, critic={critic.kind}")
    initial = [o["text"] for o in generate(policy, tok, convs, max_new_tokens=g.get("max_new_tokens", 256),
                                           temperature=g.get("temperature", 0.7), batch_size=g.get("batch_size", 16))]
    current = list(initial)
    trace = [{"prompt": p, "initial": r, "rounds": []} for p, r in zip(prompts, initial)]

    for rnd in range(cfg.get("revision_rounds", 1)):
        picked = [rng.choice(principles) for _ in prompts]
        crit_convs = [[{"role": "user", "content": p}, {"role": "assistant", "content": r},
                       {"role": "user", "content": "Critique request: " + c["critique"]}]
                      for p, r, c in zip(prompts, current, picked)]
        critiques = critic.complete(crit_convs)
        rev_convs = [cc + [{"role": "assistant", "content": cr},
                           {"role": "user", "content": "Revision request: " + c["revision"]
                            + " Reply with only the revised response to the original message."}]
                     for cc, cr, c in zip(crit_convs, critiques, picked)]
        revisions = [clean_revision(x) for x in critic.complete(rev_convs)]
        for i, (c, cr, rv) in enumerate(zip(picked, critiques, revisions)):
            trace[i]["rounds"].append({"principle": c["id"], "critique": cr, "revision": rv})
            if rv:  # keep the previous answer if the critic produced nothing usable
                current[i] = rv
        print(f"[cai] round {rnd + 1}: {sum(bool(r) for r in revisions)}/{len(revisions)} usable revisions")

    out = Path(cfg["out"]["dir"])
    write_jsonl(out / "cai_trace.jsonl", trace)
    revised = [{"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": r}], "source": "cai"}
               for p, r, t in zip(prompts, current, trace) if r and t["rounds"]]

    ratio = cfg["data"].get("helpful_ratio", 0.5)
    helpful = read_jsonl(cfg["data"]["helpful_mix"]) if cfg["data"].get("helpful_mix") else []
    k = min(len(helpful), int(len(revised) * ratio / max(1e-6, 1 - ratio)))
    mix = revised + [dict(r, source="helpful") for r in rng.sample(helpful, k)]
    rng.shuffle(mix)
    n = write_jsonl(out / "slcai_train.jsonl", mix)
    print(f"[cai] wrote {n} examples ({len(revised)} revised + {k} helpful) to {out / 'slcai_train.jsonl'}")
    print("[cai] read 20 rows of cai_trace.jsonl before training; bad revisions = bad SL-CAI model.")


if __name__ == "__main__":
    main()
