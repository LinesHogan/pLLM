"""Bounded observations of original execution identities; never a state authority."""

from __future__ import annotations

from collections import deque
from base64 import b64decode, b64encode
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from enum import StrEnum
import os
from math import prod
from time import monotonic_ns
from threading import Lock
from uuid import uuid4
from typing import Protocol

import torch
from torch._C._autograd import DeviceType

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import PhysicalPlan, TaskId
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.shared_tensor import tensor_dtype


TRACE_SCHEMA_VERSION = 5
MAX_SNAPSHOT_BYTES = 32768


class _KernelEvent(Protocol):
    name: str
    device_type: DeviceType
    device_index: int

    @property
    def device_time_total(self) -> float: ...


class TracePhase(StrEnum):
    TENSOR = "tensor"
    GRADIENT = "gradient"
    KERNEL = "kernel"
    ADMIT = "admit"
    BEGIN = "begin"
    TASK_BEGIN = "task_begin"
    TASK_END = "task_end"
    TASK_FAILED = "task_failed"
    REPLAY = "replay"
    PUBLISH = "publish"
    SAMPLER_COMPLETE = "sampler_complete"
    PREPARED = "prepared"
    SETTLED = "settled"
    ABORT = "abort"
    FAILED = "failed"
    INPUT_SKIPPED = "input_skipped"
    RETIRE = "retire"


@dataclass(frozen=True, slots=True)
class TraceEvent:
    sequence: int
    host_time_ns: int
    phase: TracePhase
    invocation_id: int
    task_id: TaskId | None = None
    identities: tuple[RequestIdentity, ...] = ()
    refs: tuple[StateRef, ...] = ()
    operation_id: int | None = None
    pin_id: int | None = None
    release_id: int | None = None
    row: PackedPhaseRow | None = None
    iteration: int | None = None
    relative_l2: float | None = None
    device_ordinal: int | None = None
    device_duration_ns: int | None = None
    tensor_name: str | None = None
    tensor_dtype: TensorDType | None = None
    tensor_shape: tuple[int, ...] = ()
    tensor_data: str | None = None
    step_id: int | None = None
    source_shape: tuple[int, ...] = ()
    kernel_name: str | None = None

    def tensor(self) -> torch.Tensor:
        """Reconstruct an independent CPU value, including exact NaN/Inf and integer bits."""
        if (self.tensor_data is None or self.tensor_dtype is None
            or len(self.tensor_shape) > 8 or any(size < 0 for size in self.tensor_shape)
            or len(self.tensor_data) > 4 * ((MAX_SNAPSHOT_BYTES + 2) // 3)):
            raise ValueError("trace event has no bounded tensor snapshot")
        dtype = tensor_dtype(self.tensor_dtype)
        raw = bytearray(b64decode(self.tensor_data, validate=True))
        if len(raw) != prod(self.tensor_shape) * dtype.itemsize or len(raw) > MAX_SNAPSHOT_BYTES:
            raise ValueError("trace tensor bytes differ from their shape and dtype")
        return (torch.frombuffer(raw, dtype=dtype).reshape(self.tensor_shape) if raw else
            torch.empty(self.tensor_shape, dtype=dtype))


@dataclass(frozen=True, slots=True)
class TraceSnapshot:
    execution_id: str
    process_id: int
    events: tuple[TraceEvent, ...]
    dropped: int
    capacity: int
    pending_device_timings: int = 0
    schema_version: int = TRACE_SCHEMA_VERSION

    def host_task_times(self) -> dict[TaskId, tuple[int, int]]:
        """Completed calls and host nanoseconds, not GPU time; unpaired/failed calls excluded."""
        starts: dict[tuple[int, TaskId], int] = {}
        totals: dict[TaskId, tuple[int, int]] = {}
        for event in self.events:
            if event.task_id is None:
                continue
            key = (event.invocation_id, event.task_id)
            if event.phase is TracePhase.TASK_BEGIN:
                starts[key] = event.host_time_ns
            elif event.phase in (TracePhase.TASK_END, TracePhase.TASK_FAILED):
                started = starts.pop(key, None)
                if started is not None and event.phase is TracePhase.TASK_END:
                    count, elapsed = totals.get(event.task_id, (0, 0))
                    totals[event.task_id] = (count + 1, elapsed + event.host_time_ns - started)
        return totals

    def device_task_times(self) -> dict[tuple[TaskId, int], tuple[int, int]]:
        """Completed stream spans per task/device; overlapping spans are not wall time."""
        totals: dict[tuple[TaskId, int], tuple[int, int]] = {}
        for event in self.events:
            if (event.phase is TracePhase.TASK_END and event.task_id is not None
                and event.device_ordinal is not None and event.device_duration_ns is not None):
                key = (event.task_id, event.device_ordinal)
                count, duration = totals.get(key, (0, 0))
                totals[key] = count + 1, duration + event.device_duration_ns
        return totals


@dataclass(frozen=True, slots=True)
class TraceSelection:
    """Explicit observations only; empty filters match all, disabled flags retain nothing.

    Tensor payloads keep at most max_tensor_bytes (a flat prefix if truncated).
    Kernel profiling uses Kineto for a finite number of task calls, synchronizes,
    and measures real device events, not host operators or CUDA stream spans.
    """

    tasks: tuple[TaskId, ...] = ()
    identities: tuple[RequestIdentity, ...] = ()
    iterations: tuple[int, ...] = ()
    names: tuple[str, ...] = ()
    tensors: bool = False
    gradients: bool = False
    max_tensor_bytes: int = MAX_SNAPSHOT_BYTES
    kernel_calls: int = 0

    def __post_init__(self) -> None:
        if (not 8 <= self.max_tensor_bytes <= MAX_SNAPSHOT_BYTES
            or not 0 <= self.kernel_calls <= 16 or any(index < 0 for index in self.iterations)):
            raise ValueError("trace selection exceeds its tensor/iteration/profiler bounds")
        if self.kernel_calls and (self.identities or self.iterations):
            raise ValueError("kernel profiling selects whole tasks; request/iteration filters apply to snapshots")


class ExecutionTrace:
    def __init__(self, plan: PhysicalPlan, capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError("trace capacity must be positive")
        self.plan = plan
        self._execution_id = uuid4().hex
        self._events: deque[tuple[TraceEvent, tuple[torch.cuda.Event, torch.cuda.Event] | None]] = deque(maxlen=capacity)
        self._timed_tasks: frozenset[TaskId] = frozenset()
        self._selection: TraceSelection = TraceSelection()
        self._kernel_calls: int = 0
        self._sequence: int = 0
        self._dropped: int = 0
        self._lock = Lock()

    def configure(self, selection: TraceSelection) -> None:
        if not set(selection.tasks).issubset(task.task_id for task in self.plan.tasks):
            raise ValueError("observation selection is outside the original physical plan")
        with self._lock:
            self._selection, self._kernel_calls = selection, selection.kernel_calls

    @contextmanager
    def private(self) -> Generator[None, None, None]:
        """Record isolated cold calls without changing the public observation log.

        The caller must own an idle execution for the whole scope. The recorder
        identity stays unchanged, including its effect namespace in Region IR.
        """
        with self._lock:
            saved = (self._events, self._sequence, self._dropped, self._timed_tasks, self._selection, self._kernel_calls)
            self._events = deque(maxlen=self._events.maxlen)
            self._sequence = self._dropped = self._kernel_calls = 0
            self._timed_tasks, self._selection = frozenset(), TraceSelection()
        try:
            yield
        finally:
            with self._lock:
                (self._events, self._sequence, self._dropped, self._timed_tasks,
                 self._selection, self._kernel_calls) = saved

    @property
    def observing(self) -> bool:
        return self._selection.tensors or self._selection.gradients

    def observe(self, invocation_id: int, task_id: TaskId, name: str, tensor: torch.Tensor, *,
                identities: tuple[RequestIdentity, ...] = (), row: PackedPhaseRow | None = None,
                refs: tuple[StateRef, ...] = (), iteration: int | None = None,
                gradient: bool = False, step_id: int | None = None) -> None:
        """Observe an explicit mathematical value or computed gradient, never install autograd hooks.

        Callers pass a single request's value when request isolation is required.
        Shared values are emitted only when every associated identity is selected.
        This copies values without retaining an autograd graph; CUDA capture is rejected.
        """
        selection = self._selection
        identities = (row.identity,) if row is not None else identities
        if (not (selection.gradients if gradient else selection.tensors)
            or selection.tasks and task_id not in selection.tasks
            or selection.names and name not in selection.names
            or selection.iterations and iteration not in selection.iterations
            or selection.identities and (not identities or not set(identities).issubset(selection.identities))):
            return
        if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("tensor/gradient observations require execution outside CUDA capture")
        shape = tuple(tensor.shape)
        if tensor.numel() * tensor.element_size() > selection.max_tensor_bytes:
            count = selection.max_tensor_bytes // tensor.element_size()
            tensor = torch.take(tensor.detach(), torch.arange(count, device=tensor.device))
        self.record(TracePhase.GRADIENT if gradient else TracePhase.TENSOR, invocation_id,
            task_id=task_id, identities=identities, refs=refs, row=row, iteration=iteration,
            tensor_name=name, tensor=tensor, step_id=step_id, source_shape=shape)

    def profile_tasks(self, tasks: tuple[TaskId, ...]) -> None:
        """Opt-in CUDA event instrumentation; an empty selection disables future timing.

        Events share the trace's capacity and oldest-drop policy. Instrumentation
        adds event launches but never waits on the device or retains user tensors.
        """
        selected = frozenset(tasks)
        if not selected.issubset(task.task_id for task in self.plan.tasks):
            raise ValueError("profiling selection is outside the original physical plan")
        with self._lock:
            self._timed_tasks = selected

    @contextmanager
    def task(self, invocation_id: int, task_id: TaskId, stream: torch.cuda.Stream | None = None) -> Generator[None]:
        with self._lock:
            timed = task_id in self._timed_tasks
            kernels = self._kernel_calls > 0 and (not self._selection.tasks or task_id in self._selection.tasks)
            if kernels:
                self._kernel_calls -= 1
        if kernels and (stream is None or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities()):
            raise RuntimeError("kernel profiling requires a CUDA task and Kineto CUDA activity support")
        if kernels and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("kernel profiling cannot start inside CUDA capture")
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
            *([torch.profiler.ProfilerActivity.CUDA] if stream is not None else [])]) if kernels else None
        timing = None
        if timed and stream is not None:
            timing = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timing[0].record(stream)
        self.record(TracePhase.TASK_BEGIN, invocation_id, task_id=task_id)
        try:
            with profiler if profiler is not None else nullcontext():
                yield
            if timing is not None:
                timing[1].record(stream)
        except Exception:
            self.record(TracePhase.TASK_FAILED, invocation_id, task_id=task_id)
            raise
        if profiler is not None:
            events = tuple[_KernelEvent, ...](profiler.events() or ())
            for event in events:
                if event.device_type == DeviceType.CUDA:
                    self.record(TracePhase.KERNEL, invocation_id, task_id=task_id,
                        kernel_name=event.name, device_ordinal=event.device_index,
                        device_duration_ns=round(event.device_time_total * 1000))
        self.record(TracePhase.TASK_END, invocation_id, task_id=task_id, timing=timing,
            device_ordinal=None if timing is None or stream is None else stream.device.index)

    def record(self, phase: TracePhase, invocation_id: int, *, task_id: TaskId | None = None,
               identities: tuple[RequestIdentity, ...] = (), refs: tuple[StateRef, ...] = (),
               operation_id: int | None = None, pin_id: int | None = None, release_id: int | None = None,
               row: PackedPhaseRow | None = None, iteration: int | None = None,
               relative_l2: float | None = None, device_ordinal: int | None = None,
               timing: tuple[torch.cuda.Event, torch.cuda.Event] | None = None,
               tensor: torch.Tensor | None = None, tensor_name: str | None = None,
               step_id: int | None = None, source_shape: tuple[int, ...] = (),
               kernel_name: str | None = None, device_duration_ns: int | None = None) -> None:
        dtype, shape, data = None, (), None
        if tensor is not None:
            if (phase not in (TracePhase.TENSOR, TracePhase.GRADIENT) or not tensor_name
                or task_id not in {task.task_id for task in self.plan.tasks}
                or tensor.ndim > 8 or tensor.numel() * tensor.element_size() > MAX_SNAPSHOT_BYTES):
                raise ValueError("tensor trace requires an original task and a bounded explicit snapshot")
            dtype = TensorDType(str(tensor.dtype).removeprefix("torch."))
            shape = tuple(tensor.shape)
            # Explicit diagnostics may synchronize; metadata-only snapshots never do.
            raw = tensor.detach().to(device="cpu").contiguous().reshape(-1).view(torch.uint8)
            data = b64encode(raw.numpy().tobytes()).decode("ascii")
        with self._lock:
            if len(self._events) == self._events.maxlen:
                self._dropped += 1
            self._sequence += 1
            self._events.append((TraceEvent(self._sequence, monotonic_ns(), phase, invocation_id, task_id,
                identities, refs, operation_id, pin_id, release_id, row, iteration, relative_l2,
                device_ordinal, tensor_name=tensor_name, tensor_dtype=dtype, tensor_shape=shape,
                tensor_data=data, step_id=step_id, source_shape=source_shape,
                kernel_name=kernel_name, device_duration_ns=device_duration_ns), timing))

    def snapshot(self) -> TraceSnapshot:
        with self._lock:
            events: list[TraceEvent] = []
            pending = 0
            for event, timing in self._events:
                if timing is not None:
                    if timing[1].query():
                        event = replace(event, device_duration_ns=round(timing[0].elapsed_time(timing[1]) * 1_000_000))
                    else:
                        pending += 1
                events.append(event)
            return TraceSnapshot(self._execution_id, os.getpid(),
                tuple(events), self._dropped, self._events.maxlen or 0, pending)
