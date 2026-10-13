"""Guardrails. Each one exists because of a Day 6 incident drill.

They run automatically inside the stage scripts; set `guardrails: false` in a
config to run a drill "unprotected" and diagnose it from symptoms instead.
"""
from __future__ import annotations

import math
import re

from .chat import IGNORE, IM_END, IM_START, template_hash
from .config import read_meta


class GuardrailError(RuntimeError):
    pass


def check_loss_mask(tok, examples: list[dict], n: int = 64) -> dict:
    """Drill 1 (prompt_loss) and drill 4 (eos_dropped).

    - No turn-opening token may carry a label (that means prompt tokens are trained on).
    - Every example with labels must label at least one end-of-turn token.
    """
    start_id = tok.convert_tokens_to_ids(IM_START)
    end_id = tok.convert_tokens_to_ids(IM_END)
    bad_prompt, missing_eos, labelled = 0, 0, 0
    for e in examples[:n]:
        lab = [l for l in e["labels"] if l != IGNORE]
        if not lab:
            continue
        labelled += 1
        if start_id in lab:
            bad_prompt += 1
        if end_id not in lab:
            missing_eos += 1
    stats = {"checked": labelled, "prompt_labelled": bad_prompt, "missing_eos": missing_eos}
    if bad_prompt:
        raise GuardrailError(f"Loss mask labels prompt/role tokens in {bad_prompt}/{labelled} examples. {stats}")
    if missing_eos > 0.5 * max(1, labelled):
        raise GuardrailError(f"End-of-turn token unlabelled in {missing_eos}/{labelled} examples; "
                             f"model will not learn to stop. {stats}")
    return stats


def check_template(parent_dir: str, tok) -> str:
    """Drill 2 (template_mismatch): a stage must render prompts exactly like its parent."""
    current = template_hash(tok)
    parent = read_meta(parent_dir).get("template_hash")
    if parent and parent != current:
        raise GuardrailError(f"Chat template hash {current} != parent checkpoint's {parent} ({parent_dir}).")
    return current


_REFUSAL = re.compile(
    r"\b(i can(?:'|no)t|i cannot|i won't|i will not|i'm (?:not able|unable)|i am (?:not able|unable)|"
    r"i'm sorry|i apologi[sz]e|as an ai|not (?:able|allowed) to (?:help|assist|provide))\b",
    re.IGNORECASE,
)


def is_refusal(text: str) -> bool:
    """Cheap heuristic; the judge-based harmlessness metric is the real one."""
    head = text[:300]
    return bool(_REFUSAL.search(head)) or not text.strip()


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)
