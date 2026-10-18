"""Instance-owned eager capture of original signed hidden resources."""

from __future__ import annotations

from dataclasses import replace
from threading import RLock

import torch
from torch.utils.hooks import RemovableHandle
from vllm import __version__ as _VLLM_VERSION

from pllm.authoring import TensorStepInput
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding, PackedHiddenSnapshot, PackedHiddenSnapshotBatch
from pllm.contracts.packed_rows import PackedPhaseRowBatch
from pllm.execution.cuda_stream import CudaStreamReadiness


class Vllm027EagerPackedHiddenCapture:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        resources: tuple[PackedHiddenResourceBinding, ...],
        plan_epoch: int,
        max_packed_rows: int,
        hidden_size: int,
    ) -> None:
        if _VLLM_VERSION != "0.27.1":
            raise RuntimeError("canonical hidden capture requires vLLM 0.27.1 exactly")
        keys = tuple(resource.locator_key for resource in resources)
        if not keys or keys != tuple(sorted(set(keys))) or min(plan_epoch, max_packed_rows, hidden_size) < 1:
            raise ValueError("signed hidden capture specialization is invalid")
        self._require_eager()
        self._resources = resources
        self._plan_epoch = plan_epoch
        self._max_packed_rows = max_packed_rows
        self._hidden_size = hidden_size
        self._row_batch: PackedPhaseRowBatch | None = None
        self._snapshots: dict[PackedHiddenResourceBinding, PackedHiddenSnapshot] = {}
        self._producer_stream: torch.cuda.Stream | None = None
        self._published = False
        self._closed = False
        self._finished_high_watermark = -1
        self._lock = RLock()
        self._bindings: dict[int, PackedHiddenResourceBinding] = {}
        self._handles: list[RemovableHandle] = []
        try:
            for resource in resources:
                module = model.get_submodule(resource.module_path)
                if id(module) in self._bindings:
                    raise ValueError("signed hidden resources alias the same model module")
                self._bindings[id(module)] = resource
                self._handles.append(
                    module.register_forward_hook(self._capture, prepend=False, with_kwargs=False, always_call=False)
                )
        except Exception:
            self.close()
            raise

    def begin(self, row_batch: PackedPhaseRowBatch) -> None:
        self._require_eager()
        with self._lock:
            if self._closed or self._row_batch is not None:
                raise RuntimeError("hidden capture is closed or already active")
            if (
                row_batch.plan_epoch != self._plan_epoch
                or row_batch.step_id <= self._finished_high_watermark
                or not 0 < row_batch.total_rows <= self._max_packed_rows
            ):
                raise RuntimeError("packed rows do not match hidden capture specialization")
            self._row_batch = row_batch

    def publish(self, value: TensorStepInput) -> TensorStepInput:
        self._require_eager()
        with self._lock:
            rows = self._require_active()
            if self._published or value.rows is not rows or value.hidden is not None:
                raise RuntimeError("hidden capture requires its original unpublished input")
            stream = self._producer_stream
            if stream is None or len(self._snapshots) != len(self._resources):
                raise RuntimeError("hidden capture did not observe every signed resource")
            batch = PackedHiddenSnapshotBatch(rows, tuple(self._snapshots[resource] for resource in self._resources))
            captured = replace(value, hidden=batch, hidden_readiness=CudaStreamReadiness(stream))
            self._published = True
            return captured

    def finish(self) -> None:
        with self._lock:
            self._require_active()
            if not self._published:
                raise RuntimeError("hidden capture must publish before finish")
            self.abort_if_active()

    def abort_if_active(self) -> None:
        with self._lock:
            if self._row_batch is not None:
                self._finished_high_watermark = self._row_batch.step_id
                self._row_batch = None
                self._snapshots.clear()
                self._producer_stream = None
                self._published = False

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.abort_if_active()
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            self._closed = True

    def _capture(
        self,
        module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        with self._lock:
            rows = self._row_batch
            if rows is None:
                return
            try:
                self._require_eager()
                resource = self._bindings[id(module)]
                if resource in self._snapshots:
                    raise RuntimeError("signed hidden resource was captured twice")
                match output:
                    case torch.Tensor() as hidden:
                        pass
                    case (torch.Tensor() as hidden, *_):
                        pass
                    case _:
                        raise RuntimeError("hidden hook requires a Tensor or Tensor-led tuple")
                if hidden.shape != (rows.total_rows, self._hidden_size) or hidden.device.type != "cuda":
                    raise RuntimeError("signed hidden output does not match packed row ABI")
                stream = torch.cuda.current_stream(hidden.device)
                if self._producer_stream is not None and stream != self._producer_stream:
                    raise RuntimeError("signed hidden resources ran on different producer streams")
                with torch.inference_mode(False):
                    owned = hidden.detach().to(dtype=torch.float32, copy=True)
                self._snapshots[resource] = PackedHiddenSnapshot(resource, owned)
                self._producer_stream = stream
            except Exception:
                self.abort_if_active()
                raise

    def _require_active(self) -> PackedPhaseRowBatch:
        if self._closed or self._row_batch is None:
            raise RuntimeError("hidden capture has no active packed step")
        return self._row_batch

    @staticmethod
    def _require_eager() -> None:
        if torch.compiler.is_compiling() or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            raise RuntimeError("canonical hidden capture is eager-only")


__all__ = ["Vllm027EagerPackedHiddenCapture"]
