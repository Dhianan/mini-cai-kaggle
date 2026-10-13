"""Drill 05 (30 min): reward-model loss and evaluation.

Interview follow-ups: why is a scalar head on the last token enough? What does
the RM's absolute scale mean (nothing)? How would you detect length bias?
"""
import torch
import torch.nn.functional as F

def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """-log P(chosen beats rejected) under Bradley-Terry, averaged."""
    raise NotImplementedError()

def pairwise_accuracy(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> float:
    raise NotImplementedError()

def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """hidden [B, T, H], right-padded attention_mask [B, T] -> hidden state of each row's last real token [B, H]."""
    raise NotImplementedError()
