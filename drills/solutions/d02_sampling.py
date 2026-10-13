"""Drill 02 (45 min): decoding strategies from raw logits.

Interview follow-ups: what does temperature do to entropy? Why apply top-k before
top-p? Why keep at least one token in top-p? What breaks at temperature 0?
"""
import torch


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """logits [B, V]. Keep the k largest per row, set the rest to -inf. k <= 0 = no-op."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set of tokens whose probability mass >= p.
    The single most likely token is always kept."""
    if p >= 1.0:
        return logits
    sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(-1)
    remove = cum - probs >= p  # mass BEFORE this token already reached p
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """logits [B, V] -> token ids [B]. temperature == 0 means greedy."""
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits / temperature
    logits = top_p_filter(top_k_filter(logits, top_k), top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
