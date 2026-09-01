"""Pure official-family TLM sentence-selection objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class TLMObjectiveResult:
    loss: torch.Tensor
    selected: torch.Tensor
    sentence_ce: torch.Tensor
    sentence_kl: torch.Tensor


def shifted_sentence_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int,
) -> torch.Tensor:
    """Compute the old TLM per-sequence shifted cross entropy exactly."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels.ne(ignore_index)
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape_as(shift_labels)
    return (token_loss * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)


def tlm_weighted_selection_objective(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    selection_logits: torch.Tensor | None,
    threshold: float,
    lamb: float,
    ignore_index: int,
) -> TLMObjectiveResult:
    """Freeze TLM's detached selector and adapted weighted update loss."""

    selector_logits = logits if selection_logits is None else selection_logits
    sentence_ce = shifted_sentence_cross_entropy(
        selector_logits,
        labels,
        ignore_index=ignore_index,
    ).detach()
    sentence_kl = shifted_sentence_cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
    )
    selected = sentence_ce > threshold
    weighted = (
        sentence_kl
        * (lamb * torch.exp(sentence_ce - threshold))
        * selected.to(sentence_kl.dtype)
    )
    selected_count = selected.sum()
    loss = torch.where(
        selected_count == 0,
        weighted.mean(),
        weighted.sum() / selected_count.clamp_min(1),
    )
    return TLMObjectiveResult(loss, selected, sentence_ce, sentence_kl)


__all__ = [
    "TLMObjectiveResult",
    "shifted_sentence_cross_entropy",
    "tlm_weighted_selection_objective",
]
