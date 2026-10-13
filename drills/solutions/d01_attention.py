"""Drill 01 (45 min): scaled dot-product attention with a causal mask, then multi-head.

Interview follow-ups: why divide by sqrt(d)? Where does the mask go and why -inf?
What changes with KV caching? With grouped-query attention?
"""
import math

import torch
import torch.nn as nn


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """q, k, v: [B, H, T, D]. Returns [B, H, T, D]. Position t may attend to <= t when causal."""
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if causal:
        T, S = q.size(-2), k.size(-2)
        mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


class MultiHeadAttention(nn.Module):
    """Self-attention over x: [B, T, d_model] -> [B, T, d_model]."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        split = lambda t: t.view(B, T, self.h, self.d).transpose(1, 2)
        y = attention(split(q), split(k), split(v), causal)
        return self.out(y.transpose(1, 2).reshape(B, T, C))
