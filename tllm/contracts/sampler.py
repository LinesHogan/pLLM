"""Closed sampler output values shared by native execution and adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.contracts.packed_rows import PackedSamplerRowBatch


@dataclass(frozen=True, slots=True)
class PackedSamplerGuidanceBatch:
    """Hidden predictions indexed in one explicit packed sampler row space."""

    sampler_rows: PackedSamplerRowBatch
    pred_hidden: torch.Tensor
    affected_row_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.pred_hidden.ndim != 2
            or int(self.pred_hidden.shape[0]) != len(self.affected_row_ids)
        ):
            raise ValueError(
                "packed sampler guidance must align predictions with affected rows"
            )
        if self.affected_row_ids != tuple(
            sorted(set(self.affected_row_ids))
        ):
            raise ValueError(
                "packed sampler guidance rows must be sorted and unique"
            )
        if any(
            row_id < 0 or row_id >= len(self.sampler_rows.links)
            for row_id in self.affected_row_ids
        ):
            raise ValueError("packed sampler guidance row is out of bounds")

    @property
    def total_rows(self) -> int:
        return len(self.sampler_rows.links)


__all__ = ["PackedSamplerGuidanceBatch"]
