"""Drill 06 (30 min): KL divergence, exactly and from samples.

Interview follow-ups: which estimators are unbiased? Which are always >= 0?
Why does RLHF use KL(pi || ref) and not KL(ref || pi)?
"""
import torch

def exact_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) for categorical distributions given logits [..., V]. Returns [...]."""
    raise NotImplementedError()

def kl_k1(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Per-sample estimate for x ~ P: log p(x) - log q(x). Unbiased, can be negative."""
    raise NotImplementedError()

def kl_k2(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """0.5 * (log p - log q)^2. Biased, >= 0, low variance when P ~ Q."""
    raise NotImplementedError()

def kl_k3(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """(r - 1) - log r with r = q(x)/p(x). Unbiased and >= 0."""
    raise NotImplementedError()
