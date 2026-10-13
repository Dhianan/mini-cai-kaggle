"""Drill 06 (30 min): KL divergence, exactly and from samples.

Interview follow-ups: which estimators are unbiased? Which are always >= 0?
Why does RLHF use KL(pi || ref) and not KL(ref || pi)?
"""
import torch


def exact_kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(P || Q) for categorical distributions given logits [..., V]. Returns [...]."""
    lp, lq = torch.log_softmax(p_logits, -1), torch.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def kl_k1(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Per-sample estimate for x ~ P: log p(x) - log q(x). Unbiased, can be negative."""
    return logp - logq


def kl_k2(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """0.5 * (log p - log q)^2. Biased, >= 0, low variance when P ~ Q."""
    return 0.5 * (logp - logq) ** 2


def kl_k3(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """(r - 1) - log r with r = q(x)/p(x). Unbiased and >= 0."""
    lr = logq - logp
    return lr.exp() - 1 - lr
