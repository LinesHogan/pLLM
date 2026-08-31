"""Backend-independent eager snapshots keyed by signed packed resources."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch

from tllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshot,
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
        self._hidden: dict[
            int,
            dict[PackedHiddenResourceBinding, _StagedPackedHidden],
        ] = {}
        self._finished_high_watermark = -1
        self._lock = RLock()

    @property
    def specialization(self) -> PackedHiddenSnapshotSpecialization:
        return self._specialization

    def stage(
        self,
        row_batch: PackedPhaseRowBatch,
        resource: PackedHiddenResourceBinding,
        layer_hidden: torch.Tensor,
    ) -> None:
        """Copy one exact packed layer tensor into service-owned storage."""

        spec = self._specialization
        with self._lock:
            if row_batch.plan_epoch != spec.plan_epoch:
                raise RuntimeError(
                    "packed hidden batch belongs to another plan epoch"
                )
            if resource not in spec.resources:
                raise RuntimeError(
                    "packed hidden resource is absent from the specialization"
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
                    "packed hidden resources must use the same row batch"
                )
            canonical_batch = (
                row_batch if staged_batch is None else staged_batch
            )
            staged_resources = self._hidden.setdefault(row_batch.step_id, {})
            if resource in staged_resources:
                raise RuntimeError("packed hidden resource is already staged")
            with torch.inference_mode(False), torch.no_grad():
                owned = layer_hidden.clone()
            self._row_batches[row_batch.step_id] = canonical_batch
            staged_resources[resource] = _StagedPackedHidden(
                row_batch=canonical_batch,
                hidden=owned,
            )

    def snapshot_for_step(
        self, plan_epoch: int, step_id: int
    ) -> PackedHiddenSnapshotBatch:
        with self._lock:
            row_batch, ordered = self._complete_step_locked(
                plan_epoch, step_id
            )
            with torch.inference_mode(False), torch.no_grad():
                snapshots = tuple(
                    PackedHiddenSnapshot(resource, staged.hidden.clone())
                    for resource, staged in zip(
                        self._specialization.resources,
                        ordered,
                        strict=True,
                    )
                )
            return PackedHiddenSnapshotBatch(
                row_batch=row_batch,
                snapshots=snapshots,
            )

    def _complete_step_locked(
        self, plan_epoch: int, step_id: int
    ) -> tuple[PackedPhaseRowBatch, tuple[_StagedPackedHidden, ...]]:
        if plan_epoch != self._specialization.plan_epoch:
            raise RuntimeError(
                "packed hidden snapshot requested from another plan epoch"
            )
        try:
            row_batch = self._row_batches[step_id]
            staged_resources = self._hidden[step_id]
        except KeyError as exc:
            raise RuntimeError(
                "packed hidden resources are not staged"
            ) from exc
        if set(staged_resources) != set(self._specialization.resources):
            raise RuntimeError(
                "not all signed packed hidden resources are staged"
            )
        ordered = tuple(
            staged_resources[resource]
            for resource in self._specialization.resources
        )
        if any(staged.row_batch != row_batch for staged in ordered):
            raise RuntimeError("packed hidden resource provenance differs")
        return row_batch, ordered

    def finish_step(self, plan_epoch: int, step_id: int) -> None:
        with self._lock:
            self._complete_step_locked(plan_epoch, step_id)
            del self._row_batches[step_id]
            del self._hidden[step_id]
            self._finished_high_watermark = max(
                self._finished_high_watermark, step_id
            )

    def abort_step(self, plan_epoch: int, step_id: int) -> None:
        """Retire a partial eager snapshot step without publishing a bundle."""

        with self._lock:
            if plan_epoch != self._specialization.plan_epoch:
                raise RuntimeError(
                    "packed hidden abort requested from another plan epoch"
                )
            if step_id not in self._row_batches:
                raise RuntimeError("packed hidden step is not staged")
            del self._row_batches[step_id]
            del self._hidden[step_id]
            self._finished_high_watermark = max(
                self._finished_high_watermark, step_id
            )


__all__ = ["PackedHiddenSnapshotService"]
