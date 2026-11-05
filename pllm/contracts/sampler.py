"""Closed sampler output values shared by native execution and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from pllm.contracts.packed_rows import PackedSamplerRowBatch


class SamplerStage(StrEnum):
    PACKED_POST_FILTER_EXACT = "packed_post_filter_exact"
    PACKED_PRE_FILTER_ADD = "packed_pre_filter_add"


@dataclass(frozen=True, slots=True)
class SamplerFeedback:
    sampler_rows: PackedSamplerRowBatch
    base_logits: torch.Tensor
    sampled_tokens: torch.Tensor


@dataclass(frozen=True, slots=True)
class PackedSamplerGuidanceBatch:
    """Hidden predictions or additive logits in the original packed row space."""

    sampler_rows: PackedSamplerRowBatch
    values: torch.Tensor
    affected_row_ids: tuple[int, ...]
    beta: float = 0.0
    stage: SamplerStage = SamplerStage.PACKED_POST_FILTER_EXACT
    member: str = ""
    order: int = 0
    # Optional per-row multiplier of beta on the device; rows with weight 0 are left unchanged.
    row_weights: torch.Tensor | None = None

    @property
    def total_rows(self) -> int:
        return len(self.sampler_rows.packed_indices)


__all__ = ["PackedSamplerGuidanceBatch", "SamplerStage", "SamplerFeedback"]
