"""Typed provenance and payloads for eager packed hidden snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from tllm.contracts.packed_rows import PackedPhaseRowBatch


class PackedHiddenRole(StrEnum):
    SOURCE = "source"
    TARGET = "target"


@dataclass(frozen=True, slots=True)
class PackedHiddenSnapshotSpecialization:
    """Cold facts for one fixed-capacity eager snapshot service."""

    plan_epoch: int
    max_packed_rows: int
    hidden_size: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        if self.plan_epoch < 1:
            raise ValueError("packed hidden plan epoch must be positive")
        if self.max_packed_rows < 1:
            raise ValueError("max packed rows must be positive")
        if self.hidden_size < 1:
            raise ValueError("packed hidden size must be positive")


@dataclass(frozen=True, slots=True)
class PackedHiddenSnapshotBatch:
    """Owned ordinary source/target tensors for one exact packed row batch."""

    row_batch: PackedPhaseRowBatch
    source_hidden: torch.Tensor
    target_hidden: torch.Tensor

    def __post_init__(self) -> None:
        if (
            self.source_hidden.ndim != 2
            or self.target_hidden.ndim != 2
            or int(self.source_hidden.shape[0]) != self.row_batch.total_rows
            or self.source_hidden.shape != self.target_hidden.shape
            or self.source_hidden.dtype is not self.target_hidden.dtype
            or self.source_hidden.device != self.target_hidden.device
        ):
            raise ValueError(
                "packed hidden source and target must align with one row batch"
            )
        if torch.is_inference(self.source_hidden) or torch.is_inference(
            self.target_hidden
        ):
            raise ValueError("packed hidden snapshots must be ordinary tensors")

    @property
    def plan_epoch(self) -> int:
        return self.row_batch.plan_epoch

    @property
    def step_id(self) -> int:
        return self.row_batch.step_id


__all__ = [
    "PackedHiddenRole",
    "PackedHiddenSnapshotBatch",
    "PackedHiddenSnapshotSpecialization",
]
