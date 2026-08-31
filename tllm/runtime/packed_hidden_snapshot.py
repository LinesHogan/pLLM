"""Backend-independent eager snapshots of packed source/target hidden rows."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch

from tllm.contracts.packed_hidden import (
    PackedHiddenRole,
    PackedHiddenSnapshotBatch,
    PackedHiddenSnapshotSpecialization,
)
from tllm.contracts.packed_rows import PackedPhaseRowBatch


@dataclass(frozen=True, slots=True)
class _StagedPackedHidden:
    row_batch: PackedPhaseRowBatch
    hidden: torch.Tensor


class PackedHiddenSnapshotService:
    """Own exact ordinary snapshots without borrowing backend decode buffers."""

    def __init__(
        self, specialization: PackedHiddenSnapshotSpecialization
    ) -> None:
        self._specialization = specialization
        self._row_batches: dict[int, PackedPhaseRowBatch] = {}
        self._source: dict[int, _StagedPackedHidden] = {}
        self._target: dict[int, _StagedPackedHidden] = {}
        self._finished_high_watermark = -1
        self._lock = RLock()

    @property
    def specialization(self) -> PackedHiddenSnapshotSpecialization:
        return self._specialization

    def stage(
        self,
        row_batch: PackedPhaseRowBatch,
        role: PackedHiddenRole,
        layer_hidden: torch.Tensor,
    ) -> None:
        """Copy one exact packed layer tensor into service-owned storage."""

        spec = self._specialization
        with self._lock:
            if row_batch.plan_epoch != spec.plan_epoch:
                raise RuntimeError(
                    "packed hidden batch belongs to another plan epoch"
                )
            staged_batch = self._row_batches.get(row_batch.step_id)
            if (
                row_batch.step_id <= self._finished_high_watermark
                and staged_batch is None
            ):
                raise RuntimeError("packed hidden step was already finished")
            if row_batch.total_rows > spec.max_packed_rows:
                raise RuntimeError(
                    "packed hidden rows exceed the specialized capacity"
                )
            expected_shape = (row_batch.total_rows, spec.hidden_size)
            if tuple(layer_hidden.shape) != expected_shape:
                raise RuntimeError(
                    "packed hidden tensor shape differs from the exact row batch"
                )
            if (
                layer_hidden.dtype is not spec.dtype
                or layer_hidden.device != spec.device
            ):
                raise RuntimeError(
                    "packed hidden tensor differs from the specialized dtype or device"
                )
            if staged_batch is not None and staged_batch != row_batch:
                raise RuntimeError(
                    "packed source and target must use the same row batch"
                )
            canonical_batch = (
                row_batch if staged_batch is None else staged_batch
            )
            match role:
                case PackedHiddenRole.SOURCE:
                    destination = self._source
                case PackedHiddenRole.TARGET:
                    destination = self._target
                case _:
                    raise AssertionError(
                        "unhandled packed hidden role"
                    )
            if row_batch.step_id in destination:
                raise RuntimeError("packed hidden role is already staged")
            with torch.inference_mode(False), torch.no_grad():
                owned = layer_hidden.clone()
            self._row_batches[row_batch.step_id] = canonical_batch
            destination[row_batch.step_id] = _StagedPackedHidden(
                row_batch=canonical_batch,
                hidden=owned,
            )

    def snapshot_for_step(
        self, plan_epoch: int, step_id: int
    ) -> PackedHiddenSnapshotBatch:
        with self._lock:
            if plan_epoch != self._specialization.plan_epoch:
                raise RuntimeError(
                    "packed hidden snapshot requested from another plan epoch"
                )
            try:
                row_batch = self._row_batches[step_id]
                source = self._source[step_id]
                target = self._target[step_id]
            except KeyError as exc:
                raise RuntimeError(
                    "packed hidden source and target are not both staged"
                ) from exc
            if (
                source.row_batch != row_batch
                or target.row_batch != row_batch
                or source.hidden.shape != target.hidden.shape
            ):
                raise RuntimeError(
                    "packed hidden source and target provenance differs"
                )
            with torch.inference_mode(False), torch.no_grad():
                source_copy = source.hidden.clone()
                target_copy = target.hidden.clone()
            return PackedHiddenSnapshotBatch(
                row_batch=row_batch,
                source_hidden=source_copy,
                target_hidden=target_copy,
            )

    def finish_step(self, plan_epoch: int, step_id: int) -> None:
        with self._lock:
            self.snapshot_for_step(plan_epoch, step_id)
            del self._row_batches[step_id]
            del self._source[step_id]
            del self._target[step_id]
            self._finished_high_watermark = max(
                self._finished_high_watermark, step_id
            )


__all__ = ["PackedHiddenSnapshotService"]
