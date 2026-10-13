"""Drill 01 (45 min): scaled dot-product attention with a causal mask, then multi-head.

Interview follow-ups: why divide by sqrt(d)? Where does the mask go and why -inf?
What changes with KV caching? With grouped-query attention?
"""
import math
import torch
import torch.nn as nn

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool=True) -> torch.Tensor:
    """q, k, v: [B, H, T, D]. Returns [B, H, T, D]. Position t may attend to <= t when causal."""
    raise NotImplementedError()

class MultiHeadAttention(nn.Module):
    """Self-attention over x: [B, T, d_model] -> [B, T, d_model]."""

    def __init__(self, d_model: int, n_heads: int):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor, causal: bool=True) -> torch.Tensor:
        raise NotImplementedError()
