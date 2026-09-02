"""Instance-owned eager CUDA capture for two signed vLLM hidden resources.

The only attachment mechanism here is two concrete ``torch`` forward hooks on
the two module paths signed in the backend attachment.  A capture instance is
owned by one in-process MRv2 runner and publishes ordinary float32 CUDA
snapshots with explicit producer-stream provenance.
"""

from __future__ import annotations

from threading import RLock

import torch
from torch.utils.hooks import RemovableHandle

from tllm.canonical.cuda_packed_hidden import CudaPackedHiddenSnapshotBatch
from tllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshot,
    PackedHiddenSnapshotBatch,
)
from tllm.contracts.packed_rows import PackedPhaseRowBatch

from vllm import __version__ as _VLLM_VERSION


_SUPPORTED_VLLM_VERSION = "0.27.1"


class Vllm027EagerPackedHiddenCapture:
    """Capture two exact signed block outputs while one packed step is open."""

    __slots__ = (
        "_closed",
        "_finished_high_watermark",
        "_hidden_size",
        "_lock",
        "_max_packed_rows",
        "_plan_epoch",
        "_producer_stream",
        "_published",
        "_row_batch",
        "_source",
        "_source_handle",
        "_source_hidden",
        "_target",
        "_target_handle",
        "_target_hidden",
    )

    _closed: bool
    _finished_high_watermark: int
    _hidden_size: int
    _max_packed_rows: int
    _plan_epoch: int
    _producer_stream: torch.cuda.Stream | None
    _published: bool
    _row_batch: PackedPhaseRowBatch | None
    _source: PackedHiddenResourceBinding
    _source_handle: RemovableHandle
    _source_hidden: torch.Tensor | None
    _target: PackedHiddenResourceBinding
    _target_handle: RemovableHandle
    _target_hidden: torch.Tensor | None

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        source: PackedHiddenResourceBinding,
        target: PackedHiddenResourceBinding,
        plan_epoch: int,
        max_packed_rows: int,
        hidden_size: int,
    ) -> None:
        if _VLLM_VERSION != _SUPPORTED_VLLM_VERSION:
            raise RuntimeError(
                "canonical hidden capture requires vLLM 0.27.1 exactly"
            )
        if (
            source.locator_key == target.locator_key
            or plan_epoch < 1
            or max_packed_rows < 1
            or hidden_size < 1
        ):
            raise ValueError("signed hidden capture specialization is invalid")
        self._require_eager()
        self._source = source
        self._target = target
        self._plan_epoch = plan_epoch
        self._max_packed_rows = max_packed_rows
        self._hidden_size = hidden_size
        self._row_batch = None
        self._source_hidden = None
        self._target_hidden = None
        self._producer_stream = None
        self._published = False
        self._finished_high_watermark = -1
        self._closed = False
        self._lock = RLock()

        source_module = model.get_submodule(source.module_path)
        target_module = model.get_submodule(target.module_path)
        source_handle = source_module.register_forward_hook(
            self._capture_source,
            prepend=False,
            with_kwargs=False,
            always_call=False,
        )
        try:
            target_handle = target_module.register_forward_hook(
                self._capture_target,
                prepend=False,
                with_kwargs=False,
                always_call=False,
            )
        except Exception:
            source_handle.remove()
            raise
        self._source_handle = source_handle
        self._target_handle = target_handle

    @property
    def plan_epoch(self) -> int:
        return self._plan_epoch

    @property
    def source(self) -> PackedHiddenResourceBinding:
        return self._source

    @property
    def target(self) -> PackedHiddenResourceBinding:
        return self._target

    def begin(self, row_batch: PackedPhaseRowBatch) -> None:
        """Open one exact packed step before the model forward begins."""

        self._require_eager()
        with self._lock:
            self._require_open()
            if (
                row_batch.plan_epoch != self._plan_epoch
                or row_batch.step_id <= self._finished_high_watermark
                or row_batch.total_rows < 1
                or row_batch.total_rows > self._max_packed_rows
            ):
                raise RuntimeError("packed rows do not match hidden capture specialization")
            if self._row_batch is not None:
                raise RuntimeError("hidden capture already has an active packed step")
            self._row_batch = row_batch
            self._source_hidden = None
            self._target_hidden = None
            self._producer_stream = None
            self._published = False

    def publish(self) -> CudaPackedHiddenSnapshotBatch:
        """Publish the two owned float32 snapshots for the active packed step."""

        self._require_eager()
        with self._lock:
            row_batch = self._require_active()
            if self._published:
                raise RuntimeError("hidden capture step was already published")
            source_hidden = self._source_hidden
            target_hidden = self._target_hidden
            producer_stream = self._producer_stream
            if (
                source_hidden is None
                or target_hidden is None
                or producer_stream is None
            ):
                raise RuntimeError("hidden capture did not observe both signed resources")
            source_snapshot = PackedHiddenSnapshot(self._source, source_hidden)
            target_snapshot = PackedHiddenSnapshot(self._target, target_hidden)
            snapshots: tuple[PackedHiddenSnapshot, PackedHiddenSnapshot]
            if self._source.locator_key < self._target.locator_key:
                snapshots = (source_snapshot, target_snapshot)
            else:
                snapshots = (target_snapshot, source_snapshot)
            batch = PackedHiddenSnapshotBatch(row_batch, snapshots)
            captured = CudaPackedHiddenSnapshotBatch.assert_produced_on(
                batch,
                producer_stream,
            )
            self._published = True
            return captured

    def finish(self) -> None:
        """Retire a published step after its consumer owns the snapshot batch."""

        with self._lock:
            row_batch = self._require_active()
            if not self._published:
                raise RuntimeError("hidden capture step must be published before finish")
            self._finished_high_watermark = max(
                self._finished_high_watermark,
                row_batch.step_id,
            )
            self._clear_active()

    def abort(self) -> None:
        """Retire an incomplete packed step after an enclosing model failure."""

        with self._lock:
            row_batch = self._require_active()
            self._finished_high_watermark = max(
                self._finished_high_watermark,
                row_batch.step_id,
            )
            self._clear_active()

    def abort_if_active(self) -> None:
        """Idempotently retire the partial step owned by this instance."""

        with self._lock:
            row_batch = self._row_batch
            if row_batch is None:
                return
            self._finished_high_watermark = max(
                self._finished_high_watermark,
                row_batch.step_id,
            )
            self._clear_active()

    def close(self) -> None:
        """Remove the two exact module hooks and retire any partial step."""

        with self._lock:
            if self._closed:
                return
            self.abort_if_active()
            self._source_handle.remove()
            self._target_handle.remove()
            self._closed = True

    def _capture_source(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        self._stage_source(output)

    def _capture_target(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        self._stage_target(output)

    def _stage_source(
        self,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        with self._lock:
            row_batch = self._row_batch
            if row_batch is None:
                return
            try:
                if self._source_hidden is not None:
                    raise RuntimeError("source hidden resource was captured twice")
                hidden, producer_stream = self._owned_float32(output, row_batch)
                self._record_producer_stream(producer_stream)
                self._source_hidden = hidden
            except Exception:
                self._abort_after_capture_failure(row_batch)
                raise

    def _stage_target(
        self,
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        with self._lock:
            row_batch = self._row_batch
            if row_batch is None:
                return
            try:
                if self._target_hidden is not None:
                    raise RuntimeError("target hidden resource was captured twice")
                hidden, producer_stream = self._owned_float32(output, row_batch)
                self._record_producer_stream(producer_stream)
                self._target_hidden = hidden
            except Exception:
                self._abort_after_capture_failure(row_batch)
                raise

    def _owned_float32(
        self,
        output: torch.Tensor | tuple[torch.Tensor, ...],
        row_batch: PackedPhaseRowBatch,
    ) -> tuple[torch.Tensor, torch.cuda.Stream]:
        self._require_eager()
        hidden = self._hidden_from_output(output)
        if (
            hidden.ndim != 2
            or int(hidden.shape[0]) != row_batch.total_rows
            or int(hidden.shape[1]) != self._hidden_size
            or hidden.device.type != "cuda"
        ):
            raise RuntimeError("signed hidden output does not match packed row ABI")
        with torch.inference_mode(False):
            owned = hidden.detach().to(dtype=torch.float32).clone()
        if torch.is_inference(owned):
            raise RuntimeError("owned hidden snapshot must not be an inference tensor")
        return owned, torch.cuda.current_stream(hidden.device)

    @staticmethod
    def _hidden_from_output(
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        match output:
            case torch.Tensor() as hidden:
                return hidden
            case (torch.Tensor() as hidden, *_):
                return hidden
            case _:
                raise RuntimeError(
                    "signed hidden hook requires a Tensor or a tuple led by Tensor"
                )

    def _record_producer_stream(self, producer_stream: torch.cuda.Stream) -> None:
        current = self._producer_stream
        if current is None:
            self._producer_stream = producer_stream
        elif current != producer_stream:
            raise RuntimeError("signed hidden resources ran on different producer streams")

    def _abort_after_capture_failure(self, row_batch: PackedPhaseRowBatch) -> None:
        if self._row_batch is row_batch:
            self._finished_high_watermark = max(
                self._finished_high_watermark,
                row_batch.step_id,
            )
            self._clear_active()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("hidden capture is closed")

    def _require_active(self) -> PackedPhaseRowBatch:
        self._require_open()
        row_batch = self._row_batch
        if row_batch is None:
            raise RuntimeError("hidden capture has no active packed step")
        return row_batch

    def _clear_active(self) -> None:
        self._row_batch = None
        self._source_hidden = None
        self._target_hidden = None
        self._producer_stream = None
        self._published = False

    @staticmethod
    def _require_eager() -> None:
        if torch.compiler.is_compiling() or (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("canonical hidden capture is eager-only")


__all__ = ["Vllm027EagerPackedHiddenCapture"]
