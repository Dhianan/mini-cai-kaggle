"""Drill 04 (30 min): the DPO loss.

Interview follow-ups: derive it from the KL-regularised RL objective. What does
beta control? Why can log p(chosen) go DOWN during training? What is IPO's fix?
"""
import torch
import torch.nn.functional as F

def dpo_loss(pi_chosen: torch.Tensor, pi_rejected: torch.Tensor, ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """All inputs [B] summed sequence log-probs. Returns (mean loss, reward accuracy)."""
    raise NotImplementedError()
