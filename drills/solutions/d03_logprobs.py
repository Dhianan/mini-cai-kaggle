"""Drill 03 (30 min): log-probability of a response given a prompt, batched and padded.

The single most common bug in post-training code is an off-by-one here.
"""
import torch


def response_logprobs(logits: torch.Tensor, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """logits [B, T, V] from a forward pass over input_ids [B, T].
    response_mask [B, T] is 1 where input_ids[b, t] is a response token we score.
    Returns sum_t log p(input_ids[b, t] | input_ids[b, <t]) over response tokens, shape [B]."""
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = input_ids[:, 1:]
    tok = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (tok * response_mask[:, 1:].to(tok.dtype)).sum(-1)
