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

    def __post_init__(self) -> None:
        count = len(self.sampler_rows.links)
        if (
            self.base_logits.ndim != 2
            or self.base_logits.shape[0] != count
            or self.sampled_tokens.shape != (count,)
            or self.sampled_tokens.dtype not in (torch.int32, torch.int64)
            or self.sampled_tokens.device != self.base_logits.device
        ):
            raise ValueError("sampler feedback must align original logits and tokens with packed rows")


@dataclass(frozen=True, slots=True)
class PackedSamplerGuidanceBatch:
    """Hidden predictions or additive logits in the original packed row space."""

    sampler_rows: PackedSamplerRowBatch
    values: torch.Tensor
    affected_row_ids: tuple[int, ...]
    beta: float = 0.0
    stage: SamplerStage = SamplerStage.PACKED_POST_FILTER_EXACT

    def __post_init__(self) -> None:
        if self.beta < 0.0 or self.values.ndim != 2 or int(self.values.shape[0]) != len(self.affected_row_ids):
            raise ValueError("packed sampler guidance must align predictions with affected rows")
        if self.affected_row_ids != tuple(sorted(set(self.affected_row_ids))):
            raise ValueError("packed sampler guidance rows must be sorted and unique")
        if any(row_id < 0 or row_id >= len(self.sampler_rows.links) for row_id in self.affected_row_ids):
            raise ValueError("packed sampler guidance row is out of bounds")

    @property
    def total_rows(self) -> int:
        return len(self.sampler_rows.links)


__all__ = ["PackedSamplerGuidanceBatch", "SamplerStage", "SamplerFeedback"]
