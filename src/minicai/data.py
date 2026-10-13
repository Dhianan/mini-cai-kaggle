"""JSONL I/O, batching, preference-pair loading, packing, and contamination checks."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

import torch

from . import faults
from .chat import IGNORE


# --------------------------------------------------------------------------- io

def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- batching

def collate(examples: list[dict], pad_id: int, padding_side: str = "right") -> dict:
    """Pad input_ids / labels and build attention_mask."""
    L = max(len(e["input_ids"]) for e in examples)
    ids, labels, attn = [], [], []
    for e in examples:
        n = len(e["input_ids"])
        pad = L - n
        lab = e.get("labels", [IGNORE] * n)
        if padding_side == "right":
            ids.append(e["input_ids"] + [pad_id] * pad)
            labels.append(lab + [IGNORE] * pad)
            attn.append([1] * n + [0] * pad)
        else:
            ids.append([pad_id] * pad + e["input_ids"])
            labels.append([IGNORE] * pad + lab)
            attn.append([0] * pad + [1] * n)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def batches(items: list, batch_size: int, shuffle: bool, seed: int, epochs: int = 1,
            start_step: int = 0) -> Iterator[tuple[int, list]]:
    """Deterministic (epoch-seeded) batches; `start_step` skips ahead for exact resume."""
    step = 0
    for ep in range(epochs):
        order = list(range(len(items)))
        if shuffle:
            random.Random(seed + ep).shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            if step >= start_step:
                yield step, [items[j] for j in order[i:i + batch_size]]
            step += 1


def pack(examples: list[dict], seq_len: int, eos_id: int) -> list[dict]:
    """Greedy sequence packing: concatenate examples (keeping their labels) into
    fixed-length rows. Cross-example attention is allowed (simple packing); use
    for SFT throughput benchmarks, not for evals."""
    rows, cur_ids, cur_lab = [], [], []
    for e in examples:
        ids, lab = e["input_ids"] + [eos_id], e["labels"] + [IGNORE]
        while ids:
            room = seq_len - len(cur_ids)
            cur_ids += ids[:room]
            cur_lab += lab[:room]
            ids, lab = ids[room:], lab[room:]
            if len(cur_ids) == seq_len:
                rows.append({"input_ids": cur_ids, "labels": cur_lab})
                cur_ids, cur_lab = [], []
    return rows


# --------------------------------------------------------------------------- preferences

def load_pairs(path: str | Path, stage: str, limit: int | None = None) -> list[dict]:
    """Load {prompt, chosen, rejected}. Fault hooks live here so DPO and the RM
    see exactly the same corruption."""
    rows = read_jsonl(path, limit)
    if stage == "rm" and faults.on("rm_length_bias"):
        for r in rows:
            if len(r["rejected"]) > len(r["chosen"]):
                r["chosen"], r["rejected"] = r["rejected"], r["chosen"]
    if faults.on("pref_swap_half"):
        rng = random.Random(0)
        for r in rows:
            if rng.random() < 0.5:
                r["chosen"], r["rejected"] = r["rejected"], r["chosen"]
    return rows


# --------------------------------------------------------------------------- contamination

_WS = re.compile(r"\s+")


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _WS.sub(" ", text.lower()).strip().split(" ")
    if len(toks) < n:
        return {tuple(toks)} if toks and toks[0] else set()
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def contamination(train_texts: Iterable[str], eval_texts: Iterable[str], n: int = 8) -> dict:
    """Fraction of eval items sharing at least one n-gram with training data."""
    train_grams: set = set()
    for t in train_texts:
        train_grams |= _ngrams(t, n)
    eval_texts = list(eval_texts)
    hits = [t for t in eval_texts if _ngrams(t, n) & train_grams]
    return {"n": n, "eval_items": len(eval_texts), "contaminated": len(hits),
            "rate": len(hits) / max(1, len(eval_texts)), "examples": hits[:3]}


def decontaminate(train_rows: list[dict], text_fn, eval_texts: Iterable[str], n: int = 8) -> list[dict]:
    eval_grams: set = set()
    for t in eval_texts:
        eval_grams |= _ngrams(t, n)
    return [r for r in train_rows if not (_ngrams(text_fn(r), n) & eval_grams)]
