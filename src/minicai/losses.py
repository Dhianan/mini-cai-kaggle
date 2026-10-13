"""Hand-written post-training losses. Every function here is unit-tested against
an independent reference (tests/test_losses.py) and is what interviewers ask
you to write on a whiteboard.

Shapes: B = batch, T = sequence length, V = vocab, G = group size.
Labels use -100 for "no loss here".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = -100


# --------------------------------------------------------------------------- SFT

def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy averaged over labelled tokens only.

    logits[:, t] predicts token t+1, so shift by one. Averaging per *token*
    (not per sequence) is the standard choice; note it weights long answers more.
    """
    logits = logits[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)).float(), labels.view(-1), ignore_index=IGNORE
    )


# --------------------------------------------------------------------------- log-probs

def token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log p(label) and the loss mask, both [B, T-1]."""
    logits = logits[:, :-1].float()
    labels = labels[:, 1:]
    mask = labels != IGNORE
    safe = labels.masked_fill(~mask, 0)
    logps = torch.log_softmax(logits, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return logps * mask, mask


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor, average: bool = False) -> torch.Tensor:
    """Sum (or mean) of log p over labelled tokens, one number per sequence [B]."""
    logps, mask = token_logprobs(logits, labels)
    total = logps.sum(-1)
    return total / mask.sum(-1).clamp(min=1) if average else total


def token_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean predictive entropy over labelled positions (a collapse detector)."""
    logits = logits[:, :-1].float()
    mask = labels[:, 1:] != IGNORE
    logp = torch.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    return (ent * mask).sum() / mask.sum().clamp(min=1)


# --------------------------------------------------------------------------- reward model

def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """-log sigmoid(r_c - r_r). P(chosen > rejected) = sigmoid(r_c - r_r)."""
    margin = r_chosen - r_rejected
    loss = -F.logsigmoid(margin).mean()
    return loss, {
        "acc": (margin > 0).float().mean(),
        "margin": margin.mean(),
        "r_chosen": r_chosen.mean(),
        "r_rejected": r_rejected.mean(),
    }


# --------------------------------------------------------------------------- DPO

def dpo_loss(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Direct Preference Optimization (Rafailov et al., 2023).

    Implicit reward r(x, y) = beta * (log pi(y|x) - log ref(y|x)).
    loss = -log sigmoid(r(x, y_c) - r(x, y_r)).
    label_smoothing > 0 gives conservative DPO (robust to noisy labels).
    All inputs are summed sequence log-probs, shape [B].
    """
    chosen_r = beta * (pi_chosen - ref_chosen)
    rejected_r = beta * (pi_rejected - ref_rejected)
    logits = chosen_r - rejected_r
    loss = -(1 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)
    return loss.mean(), {
        "reward_acc": (logits > 0).float().mean(),
        "reward_margin": logits.mean(),
        "reward_chosen": chosen_r.mean(),
        "reward_rejected": rejected_r.mean(),
        "logp_chosen": pi_chosen.mean(),
        "logp_rejected": pi_rejected.mean(),
    }


# --------------------------------------------------------------------------- KL

def kl_k1(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Naive per-token estimator log(pi/ref) on samples from pi. Unbiased, high
    variance, can be negative per token."""
    return logp - ref_logp


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Schulman's k3: (r - 1) - log r with r = ref/pi. Unbiased, always >= 0,
    low variance. Used as the RL penalty."""
    log_ratio = ref_logp - logp
    return torch.exp(log_ratio) - 1 - log_ratio


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    mask = mask.to(x.dtype)
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)


# --------------------------------------------------------------------------- advantages

def gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float = 1.0, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over a single trajectory per row.

    rewards, values: [B, T]. Value after the last step is taken as 0.
    Returns (advantages, returns), both [B, T].
    """
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(B, dtype=rewards.dtype, device=rewards.device)
    for t in reversed(range(T)):
        next_v = values[:, t + 1] if t + 1 < T else torch.zeros_like(last)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + values


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6, scale: bool = True) -> torch.Tensor:
    """GRPO: normalise each reward against its own group of G samples for the
    same prompt. rewards [B, G] -> advantages [B, G]."""
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if not scale:
        return centered
    return centered / (rewards.std(dim=1, keepdim=True) + eps)


def clipped_pg_loss(
    logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor, clip: float = 0.2
) -> tuple[torch.Tensor, dict]:
    """PPO clipped surrogate, token level. adv is [B] (sequence) or [B, T]."""
    if adv.dim() == 1:
        adv = adv.unsqueeze(1).expand_as(logp)
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    per_tok = -torch.minimum(unclipped, clipped)
    loss = masked_mean(per_tok, mask)
    clip_frac = masked_mean(((ratio - 1).abs() > clip).float(), mask)
    return loss, {"clip_frac": clip_frac, "ratio_mean": masked_mean(ratio, mask)}
