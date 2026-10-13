"""Critic / labeler / judge backends behind one interface.

kind: "self"      -> reuse the policy being trained (pure CAI; weak at 1.5B)
      "hf"        -> a local open model, e.g. Qwen/Qwen2.5-7B-Instruct
      "anthropic" -> Claude via API (needs ANTHROPIC_API_KEY); best label quality

Pairwise judgments are always run in BOTH orders to cancel position bias, and
the disagreement rate between orders is reported as a labeler-quality metric.
"""
from __future__ import annotations

import os
import time

import torch

from .chat import prompt_text
from .models import generate, load_policy

PAIR_TEMPLATE = """Consider this conversation between a human and an AI assistant.

Human: {prompt}

Which assistant response is better according to this principle?
Principle: {principle}

Response (A):
{a}

Response (B):
{b}

Answer with a single letter, A or B."""

YESNO_SUFFIX = "\n\nAnswer with a single word, Yes or No."


class Judge:
    def __init__(self, cfg: dict, policy=None, tok=None):
        self.kind = cfg.get("kind", "self")
        self.max_new_tokens = cfg.get("max_new_tokens", 384)
        self.temperature = cfg.get("temperature", 0.7)
        self.batch_size = cfg.get("batch_size", 8)
        if self.kind == "self":
            assert policy is not None, "judge kind 'self' needs the policy model"
            self.model, self.tok = policy, tok
        elif self.kind == "hf":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.tok = load_policy(cfg["model"], dtype=cfg.get("dtype", "auto"), device=device)
            self.model.eval()
        elif self.kind == "anthropic":
            import anthropic  # pip install anthropic

            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model_name = cfg.get("model", "claude-haiku-5-5")
        else:
            raise ValueError(f"Unknown judge kind {self.kind!r}")

    # ------------------------------------------------------------------ free text
    def complete(self, conversations: list[list[dict]]) -> list[str]:
        if self.kind == "anthropic":
            return [self._api(c, self.max_new_tokens) for c in conversations]
        outs = generate(self.model, self.tok, conversations, max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature, batch_size=self.batch_size)
        return [o["text"] for o in outs]

    # ------------------------------------------------------------------ choices
    @torch.no_grad()
    def _choice_probs(self, questions: list[str], options: tuple[str, str]) -> list[float]:
        """P(option[0]) vs option[1] for each question."""
        if self.kind == "anthropic":
            probs = []
            for q in questions:
                ans = self._api([{"role": "user", "content": q}], 5).strip().lower()
                probs.append(1.0 if ans.startswith(options[0].lower()) else
                             0.0 if ans.startswith(options[1].lower()) else 0.5)
            return probs
        tok, model = self.tok, self.model
        device = next(model.parameters()).device
        ids = [tok(o, add_special_tokens=False)["input_ids"][0] for o in options]
        old = tok.padding_side
        tok.padding_side = "left"
        probs = []
        try:
            for i in range(0, len(questions), self.batch_size):
                chunk = [prompt_text([{"role": "user", "content": q}]) for q in questions[i:i + self.batch_size]]
                enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                logits = model(**enc).logits[:, -1, ids].float()
                probs += torch.softmax(logits, -1)[:, 0].tolist()
        finally:
            tok.padding_side = old
        return probs

    def prefer(self, prompts: list[str], a: list[str], b: list[str], principles: list[str]) -> list[dict]:
        """Returns per item {p_a, consistent}; p_a is averaged over both orders."""
        fwd = [PAIR_TEMPLATE.format(prompt=p, principle=c, a=x, b=y) for p, x, y, c in zip(prompts, a, b, principles)]
        rev = [PAIR_TEMPLATE.format(prompt=p, principle=c, a=y, b=x) for p, x, y, c in zip(prompts, a, b, principles)]
        pf = self._choice_probs(fwd, ("A", "B"))
        pr = self._choice_probs(rev, ("A", "B"))
        out = []
        for f, r in zip(pf, pr):
            p_a = (f + (1 - r)) / 2
            out.append({"p_a": p_a, "consistent": (f > 0.5) == ((1 - r) > 0.5), "p_fwd": f, "p_rev": 1 - r})
        return out

    def yes_prob(self, questions: list[str]) -> list[float]:
        return self._choice_probs([q + YESNO_SUFFIX for q in questions], ("Yes", "No"))

    # ------------------------------------------------------------------ api
    def _api(self, messages: list[dict], max_tokens: int) -> str:
        for attempt in range(5):
            try:
                r = self.client.messages.create(model=self.model_name, max_tokens=max_tokens, messages=messages,
                                                temperature=0.0 if max_tokens <= 5 else self.temperature)
                return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            except Exception as e:
                wait = 2 ** attempt
                print(f"[judge] API error ({e}); retry in {wait}s")
                time.sleep(wait)
        return ""
