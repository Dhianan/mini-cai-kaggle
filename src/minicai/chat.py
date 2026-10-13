"""Chat template, tokenization with assistant-only loss masks, and template hashing.

We render the template by hand instead of via a Jinja chat_template so the
loss mask is exact by construction: each segment is tokenized separately and
we know which token ids came from assistant content.

Format (ChatML, the same tokens Qwen2.5 uses):

    <|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n{content}<|im_end|>\n

Loss is taken on assistant *content + <|im_end|>*. Including <|im_end|> is what
teaches the model to stop (see fault `eos_dropped`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import faults

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
IGNORE = -100

_STAGE = {"name": "sft"}

# Same format as a Jinja template, set on every tokenizer we load/save so library
# code (TRL, vLLM, HF pipelines) renders conversations identically to us.
CHATML_JINJA = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@dataclass(frozen=True)
class Template:
    name: str
    turn_open: str  # format string with {role}
    turn_close: str

    def open(self, role: str) -> str:
        return self.turn_open.format(role=role)


CHATML = Template("chatml", IM_START + "{role}\n", IM_END + "\n")
# Plausible-looking variant used only by the `template_mismatch` fault.
_ALT = Template("chatml-alt", IM_START + "{role}:\n", IM_END + "\n\n")


def set_stage(name: str) -> None:
    _STAGE["name"] = name


def current_template() -> Template:
    if faults.on("template_mismatch") and _STAGE["name"] in ("dpo", "rl"):
        return _ALT
    return CHATML


def template_hash(tokenizer=None) -> str:
    t = current_template()
    parts = [t.turn_open, t.turn_close]
    if tokenizer is not None:
        parts += [str(tokenizer.convert_tokens_to_ids(IM_START)), str(tokenizer.convert_tokens_to_ids(IM_END))]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def _ids(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def encode_chat(tok, messages: list[dict], max_len: int | None = None) -> dict:
    """Tokenize a full conversation. labels == input_ids on assistant content + end
    token, IGNORE elsewhere."""
    t = current_template()
    input_ids: list[int] = []
    labels: list[int] = []
    prompt_loss = faults.on("prompt_loss")
    drop_eos = faults.on("eos_dropped")
    for m in messages:
        head = _ids(tok, t.open(m["role"]))
        body = _ids(tok, m["content"])
        end = _ids(tok, IM_END)
        nl = _ids(tok, t.turn_close[len(IM_END):])
        is_asst = m["role"] == "assistant"
        if is_asst and drop_eos:
            end_label = [IGNORE] * len(end)
        elif is_asst:
            end_label = end
        else:
            end_label = [IGNORE] * len(end)
        seg_ids = head + body + end + nl
        if is_asst:
            seg_labels = [IGNORE] * len(head) + body + end_label + [IGNORE] * len(nl)
        else:
            seg_labels = [IGNORE] * len(seg_ids)
        if prompt_loss:
            seg_labels = list(seg_ids)
        input_ids += seg_ids
        labels += seg_labels
    if max_len is not None:
        input_ids, labels = input_ids[:max_len], labels[:max_len]
    return {"input_ids": input_ids, "labels": labels}


def prompt_text(messages: list[dict]) -> str:
    """Rendered prompt ending with an open assistant turn, ready for generation."""
    t = current_template()
    s = "".join(t.open(m["role"]) + m["content"] + t.turn_close for m in messages)
    return s + t.open("assistant")


def encode_prompt_response(tok, messages: list[dict], response: str, max_len: int | None = None) -> dict:
    """Tokenize prompt + one assistant response; labels only on the response.
    Used by RM, DPO, and RL so all three score exactly the same tokens."""
    return encode_chat(tok, list(messages) + [{"role": "assistant", "content": response}], max_len)


def as_messages(prompt: str | list[dict]) -> list[dict]:
    return [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
