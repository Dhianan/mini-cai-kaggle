"""Drill 02 (45 min): decoding strategies from raw logits.

Interview follow-ups: what does temperature do to entropy? Why apply top-k before
top-p? Why keep at least one token in top-p? What breaks at temperature 0?
"""
import torch

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """logits [B, V]. Keep the k largest per row, set the rest to -inf. k <= 0 = no-op."""
    raise NotImplementedError()

def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set of tokens whose probability mass >= p.
The single most likely token is always kept."""
    raise NotImplementedError()

def sample(logits: torch.Tensor, temperature: float=1.0, top_k: int=0, top_p: float=1.0, generator: torch.Generator | None=None) -> torch.Tensor:
    """logits [B, V] -> token ids [B]. temperature == 0 means greedy."""
    raise NotImplementedError()
