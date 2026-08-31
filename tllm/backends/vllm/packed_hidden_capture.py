"""Runner-owned eager capture of signed packed block-output resources."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

import torch
from torch.utils.hooks import RemovableHandle

from tllm.backends.protocol import (
    PackedHiddenSnapshotSink,
    RunnerBinding,
)
from tllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshotBatch,
    PackedHiddenSnapshotSpecialization,
)
from tllm.contracts.packed_rows import PackedPhaseRowBatch
from tllm.runtime.packed_hidden_snapshot import PackedHiddenSnapshotService
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.workgraph.graph_mode import GraphMode


class PackedHiddenExecution(PackedHiddenSnapshotSink, Protocol):
    @property
    def plan_epoch(self) -> int: ...

    @property
    def requires_packed_hidden(self) -> bool: ...

    @property
    def packed_hidden_resources(
        self,
    ) -> tuple[PackedHiddenResourceBinding, ...]: ...


def _exact_module_path_tokens(path: str) -> tuple[str | int, ...]:
    tokens: list[str | int] = []
    for component in path.split("."):
        attribute, separator, indices = component.partition("[")
        if not attribute:
            raise RuntimeError("packed hidden module_path has an empty component")
        tokens.append(attribute)
        remainder = "" if not separator else "[" + indices
        while remainder:
            if not remainder.startswith("["):
                raise RuntimeError("packed hidden module_path index is malformed")
            close = remainder.find("]")
            if close < 0:
                raise RuntimeError("packed hidden module_path index is malformed")
            index = remainder[1:close]
            if not index or not index.lstrip("-").isdigit():
                raise RuntimeError("packed hidden module_path index is malformed")
            tokens.append(int(index))
            remainder = remainder[close + 1 :]
    return tuple(tokens)


def _resolve_exact_module(
    model: torch.nn.Module, path: str
) -> torch.nn.Module:
    module = model
    for token in _exact_module_path_tokens(path):
        match token:
            case str() as attribute:
                module = module.get_submodule(attribute)
            case int() as index:
                match module:
                    case torch.nn.ModuleList() as indexed:
                        module = indexed[index]
                    case torch.nn.Sequential() as indexed:
                        module = indexed[index]
                    case _:
                        raise RuntimeError(
                            "packed hidden bracket index requires an exact "
                            "ModuleList or Sequential container"
                        )
    return module


_ForwardHook = Callable[
    [
        torch.nn.Module,
        tuple[object, ...],
        torch.Tensor | tuple[torch.Tensor, ...],
    ],
    None,
]


@dataclass(slots=True)
class _ActivePackedCapture:
    row_batch: PackedPhaseRowBatch
    service_started: bool = False
    published: bool = False


def _require_eager() -> None:
    if torch.compiler.is_compiling():
        raise RuntimeError("packed hidden capture is eager-only")
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("packed hidden capture is eager-only")


def _hidden_from_forward_output(
    output: torch.Tensor | tuple[torch.Tensor, ...],
) -> torch.Tensor:
    match output:
        case torch.Tensor() as hidden:
            return hidden
        case (torch.Tensor() as hidden, *_):
            return hidden
        case _:
            raise RuntimeError(
                "packed hidden capture requires a Tensor output or a tuple "
                "whose first element is a Tensor"
            )


class RunnerPackedHiddenCapture:
    """Capture exact signed module outputs only while a packed step is active."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        resources: tuple[PackedHiddenResourceBinding, ...],
        specialization: RuntimeSpecializationSpec,
        plan_epoch: int,
    ) -> None:
        max_packed_rows = max(
            specialization.rows,
            (
                specialization.max_prefill_tokens
                if specialization.max_prefill_tokens is not None
                else specialization.rows
            ),
        )
        self._resources = resources
        self._service = PackedHiddenSnapshotService(
            PackedHiddenSnapshotSpecialization(
                plan_epoch=plan_epoch,
                max_packed_rows=max_packed_rows,
                hidden_size=specialization.hidden_size,
                resources=resources,
                dtype=specialization.dtype,
                device=specialization.device,
            )
        )
        grouped: list[
            tuple[
                torch.nn.Module,
                list[PackedHiddenResourceBinding],
            ]
        ] = []
        for resource in resources:
            module = _resolve_exact_module(model, resource.module_path)
            for grouped_module, grouped_resources in grouped:
                if grouped_module is module:
                    grouped_resources.append(resource)
                    break
            else:
                grouped.append((module, [resource]))

        handles: list[RemovableHandle] = []
        for module, grouped_resources in grouped:
            exact_resources = tuple(grouped_resources)
            handle = module.register_forward_hook(
                self._hook_for_resources(exact_resources)
            )
            handles.append(handle)
        self._handles = tuple(handles)
        self._active: _ActivePackedCapture | None = None
        self._finished_high_watermark = -1
        self._closed = False
        self._lock = RLock()

    @property
    def plan_epoch(self) -> int:
        return self._service.specialization.plan_epoch

    @property
    def resources(self) -> tuple[PackedHiddenResourceBinding, ...]:
        return self._resources

    def begin(self, row_batch: PackedPhaseRowBatch) -> None:
        """Make one exact packed batch visible to installed eager hooks."""

        _require_eager()
        with self._lock:
            if self._closed:
                raise RuntimeError("packed hidden capture is closed")
            if row_batch.plan_epoch != self._service.specialization.plan_epoch:
                raise RuntimeError("packed hidden rows belong to another plan epoch")
            if row_batch.step_id <= self._finished_high_watermark:
                raise RuntimeError("packed hidden step was already finished")
            if self._active is not None:
                raise RuntimeError("packed hidden capture already has an active step")
            active = _ActivePackedCapture(row_batch)
            self._active = active
            if row_batch.total_rows == 0:
                specialization = self._service.specialization
                empty = torch.empty(
                    (0, specialization.hidden_size),
                    dtype=specialization.dtype,
                    device=specialization.device,
                )
                for resource in specialization.resources:
                    self._service.stage(row_batch, resource, empty)
                active.service_started = True

    def publish(
        self,
        execution: PackedHiddenSnapshotSink | None = None,
    ) -> PackedHiddenSnapshotBatch:
        """Return one complete resource-keyed snapshot batch for the active step."""

        _require_eager()
        with self._lock:
            active = self._require_active()
            if active.published:
                raise RuntimeError("packed hidden step was already published")
            try:
                snapshot = self._service.snapshot_for_step(
                    active.row_batch.plan_epoch,
                    active.row_batch.step_id,
                )
                if execution is not None:
                    execution.stage_packed_hidden_snapshots(snapshot)
            except Exception:
                self._abort_active_locked()
                raise
            active.published = True
            return snapshot

    def finish(self) -> None:
        """Retire a published capture batch and advance its ABA watermark."""

        with self._lock:
            active = self._require_active()
            if not active.published:
                raise RuntimeError("packed hidden step must be published before finish")
            try:
                self._service.finish_step(
                    active.row_batch.plan_epoch,
                    active.row_batch.step_id,
                )
            finally:
                self._finished_high_watermark = max(
                    self._finished_high_watermark,
                    active.row_batch.step_id,
                )
                self._active = None

    def abort(self) -> None:
        """Retire the active partial capture without publishing it."""

        with self._lock:
            self._require_active()
            self._abort_active_locked()

    def abort_if_active(self) -> None:
        """Abort a partial runner step when an enclosing forward fails."""

        with self._lock:
            if self._active is None:
                return
            self._abort_active_locked()

    def close(self) -> None:
        """Remove all runner module hooks and retire an active partial batch."""

        with self._lock:
            if self._closed:
                return
            try:
                if self._active is not None:
                    self._abort_active_locked()
            finally:
                for handle in self._handles:
                    handle.remove()
                self._closed = True

    def _hook_for_resources(
        self,
        resources: tuple[PackedHiddenResourceBinding, ...],
    ) -> _ForwardHook:
        def capture(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            self._capture_resources(resources, output)

        return capture

    def _capture_resources(
        self,
        resources: tuple[PackedHiddenResourceBinding, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        with self._lock:
            active = self._active
            if active is None:
                return
            try:
                _require_eager()
                if active.published:
                    raise RuntimeError("packed hidden step was already published")
                hidden = _hidden_from_forward_output(output)
                if hidden.ndim != 2:
                    raise RuntimeError("packed hidden capture requires rank-2 output")
                if int(hidden.shape[0]) < active.row_batch.total_rows:
                    raise RuntimeError(
                        "packed hidden output has fewer rows than the active batch"
                    )
                # PackedPhaseRowBatch signs the canonical contiguous row order.
                selected = hidden[: active.row_batch.total_rows]
                for resource in resources:
                    self._service.stage(active.row_batch, resource, selected)
                    active.service_started = True
            except Exception:
                self._abort_active_locked()
                raise

    def _require_active(self) -> _ActivePackedCapture:
        if self._closed:
            raise RuntimeError("packed hidden capture is closed")
        if self._active is None:
            raise RuntimeError("packed hidden capture has no active step")
        return self._active

    def _abort_active_locked(self) -> None:
        active = self._active
        if active is None:
            return
        try:
            if active.service_started:
                self._service.abort_step(
                    active.row_batch.plan_epoch,
                    active.row_batch.step_id,
                )
        finally:
            self._finished_high_watermark = max(
                self._finished_high_watermark,
                active.row_batch.step_id,
            )
            self._active = None


def begin_runner_packed_hidden_capture(
    *,
    binding: RunnerBinding | None,
    model: torch.nn.Module,
    execution: PackedHiddenExecution | None,
    specialization: RuntimeSpecializationSpec,
    graph_mode: GraphMode,
    row_batch: PackedPhaseRowBatch,
) -> None:
    """Install or refresh one runner-owned eager capture and open its step."""

    if execution is None or not execution.requires_packed_hidden:
        return
    if binding is None:
        raise RuntimeError(
            "packed hidden execution requires a bound vLLM runner"
        )
    if graph_mode is not GraphMode.EAGER:
        raise RuntimeError("packed hidden capture requires eager graph mode")
    resources = execution.packed_hidden_resources
    synchronize_runner_packed_hidden_capture(
        binding=binding,
        execution=execution,
    )
    capture = binding.packed_hidden_capture
    if capture is None:
        capture = RunnerPackedHiddenCapture(
            model=model,
            resources=resources,
            specialization=specialization,
            plan_epoch=execution.plan_epoch,
        )
        binding.packed_hidden_capture = capture
    capture.begin(row_batch)


def synchronize_runner_packed_hidden_capture(
    *,
    binding: RunnerBinding | None,
    execution: PackedHiddenExecution | None,
) -> None:
    """Close a runner hook whose signed packed topology is no longer active."""

    if binding is None or binding.packed_hidden_capture is None:
        return
    capture = binding.packed_hidden_capture
    if (
        execution is not None
        and execution.requires_packed_hidden
        and capture.plan_epoch == execution.plan_epoch
        and capture.resources == execution.packed_hidden_resources
    ):
        return
    close_runner_packed_hidden_capture(binding)


def publish_runner_packed_hidden_capture(
    *,
    binding: RunnerBinding | None,
    execution: PackedHiddenExecution | None,
) -> None:
    """Publish the active runner capture before the signed model-post event."""

    if execution is None or not execution.requires_packed_hidden:
        return
    if binding is None or binding.packed_hidden_capture is None:
        raise RuntimeError("packed hidden execution has no runner capture")
    capture = binding.packed_hidden_capture
    capture.publish(execution)
    capture.finish()


def abort_runner_packed_hidden_capture(binding: RunnerBinding | None) -> None:
    if binding is None or binding.packed_hidden_capture is None:
        return
    binding.packed_hidden_capture.abort_if_active()


def close_runner_packed_hidden_capture(binding: RunnerBinding) -> None:
    capture = binding.packed_hidden_capture
    if capture is None:
        return
    capture.close()
    binding.packed_hidden_capture = None


__all__ = [
    "PackedHiddenExecution",
    "RunnerPackedHiddenCapture",
    "abort_runner_packed_hidden_capture",
    "begin_runner_packed_hidden_capture",
    "close_runner_packed_hidden_capture",
    "publish_runner_packed_hidden_capture",
    "synchronize_runner_packed_hidden_capture",
]
