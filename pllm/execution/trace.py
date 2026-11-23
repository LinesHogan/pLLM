"""A bounded trace of one execution: task timings, observed tensors and exports, keyed by execution identity.

StateExecution and TensorOperation record into ExecutionTrace; callers read it through snapshots. It never
decides state.
"""

from __future__ import annotations

import atexit
import os
from base64 import b64decode, b64encode
from collections import deque
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from math import prod
from threading import Condition, Thread
from time import monotonic_ns
from typing import Literal, Protocol
from uuid import uuid4

import torch
from torch._C._autograd import DeviceType

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import ScheduleDecision, TaskId
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.admission import WindowBackpressure

TRACE_SCHEMA_VERSION = 10
MAX_SNAPSHOT_BYTES = 32768
# Events an execution trace retains; pending exports are never evicted, so this bounds their host copies.
EVENT_CAPACITY = 256


class _KernelEvent(Protocol):
    name: str
    device_type: DeviceType
    device_index: int

    @property
    def device_time_total(self) -> float: ...


class TraceEventKind(StrEnum):
    TENSOR = "tensor"
    GRADIENT = "gradient"
    KERNEL = "kernel"
    ADMIT = "admit"
    BEGIN = "begin"
    TASK_BEGIN = "task_begin"
    TASK_END = "task_end"
    TASK_FAILED = "task_failed"
    BACKPRESSURE = "backpressure"
    CUSTOM_FORWARD = "custom_forward"
    PUBLISH = "publish"
    SAMPLER_COMPLETE = "sampler_complete"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORT = "abort"
    FAILED = "failed"
    INPUT_SKIPPED = "input_skipped"
    RETIRE = "retire"


@dataclass(frozen=True, slots=True)
class TraceEvent:
    sequence: int
    host_time_ns: int
    kind: TraceEventKind
    invocation_id: int
    task_id: TaskId | None = None
    identities: tuple[RequestIdentity, ...] = ()
    refs: tuple[StateRef, ...] = ()
    # An export row's token position within its request, beside its identity.
    positions: tuple[int, ...] = ()
    # Each exported row's resolved physical layer index when reading a single LayerRoute (-1 when unmatched).
    layers: tuple[int, ...] = ()
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
    dropped_rows: int = 0
    kernel_name: str | None = None

    def tensor(self) -> torch.Tensor:
        """Reconstruct an independent CPU value, including exact NaN/Inf and integer bits."""
        if self.tensor_data is None or self.tensor_dtype is None:
            raise ValueError("trace event has no bounded tensor snapshot")
        dtype = self.tensor_dtype.torch
        raw = bytearray(b64decode(self.tensor_data, validate=True))
        assert len(raw) == prod(self.tensor_shape) * dtype.itemsize <= MAX_SNAPSHOT_BYTES
        return (
            torch.frombuffer(raw, dtype=dtype).reshape(self.tensor_shape)
            if raw
            else torch.empty(self.tensor_shape, dtype=dtype)
        )


@dataclass(frozen=True, slots=True)
class TraceSnapshot:
    execution_id: str
    process_id: int
    events: tuple[TraceEvent, ...]
    dropped: int
    capacity: int
    pending_device_timings: int = 0
    export_dropped: int = 0
    schema_version: int = TRACE_SCHEMA_VERSION

    def host_task_times(self) -> dict[TaskId, tuple[int, int]]:
        """Completed calls and host nanoseconds, not GPU time; unpaired/failed calls excluded."""
        starts: dict[tuple[int, TaskId], int] = {}
        totals: dict[TaskId, tuple[int, int]] = {}
        for event in self.events:
            if event.task_id is None:
                continue
            key = (event.invocation_id, event.task_id)
            if event.kind is TraceEventKind.TASK_BEGIN:
                starts[key] = event.host_time_ns
            elif event.kind in (TraceEventKind.TASK_END, TraceEventKind.TASK_FAILED):
                started = starts.pop(key, None)
                if started is not None and event.kind is TraceEventKind.TASK_END:
                    count, elapsed = totals.get(event.task_id, (0, 0))
                    totals[event.task_id] = (count + 1, elapsed + event.host_time_ns - started)
        return totals

    def device_task_times(self) -> dict[tuple[TaskId, int], tuple[int, int]]:
        """Completed stream spans per task/device; overlapping spans are not wall time."""
        totals: dict[tuple[TaskId, int], tuple[int, int]] = {}
        for event in self.events:
            if (
                event.kind is TraceEventKind.TASK_END
                and event.task_id is not None
                and event.device_ordinal is not None
                and event.device_duration_ns is not None
            ):
                key = (event.task_id, event.device_ordinal)
                count, duration = totals.get(key, (0, 0))
                totals[key] = count + 1, duration + event.device_duration_ns
        return totals


@dataclass(frozen=True, slots=True)
class TraceSelection:
    """Explicit observations only; empty filters match all, disabled flags retain nothing.

    Tensor payloads keep at most max_tensor_bytes (a flat prefix if truncated).
    `states` records each retiring request's newest state fields as RETIRE events (testing.check_on_gpu compares them).
    Kernel profiling uses Kineto for a finite number of task calls, synchronizes,
    and measures real device events, not host operators or CUDA stream spans.
    """

    tasks: tuple[TaskId, ...] = ()
    identities: tuple[RequestIdentity, ...] = ()
    iterations: tuple[int, ...] = ()
    names: tuple[str, ...] = ()
    tensors: bool = False
    gradients: bool = False
    states: bool = False
    max_events: int = 0  # Zero keeps the recorder's current bounded capacity.
    max_tensor_bytes: int = MAX_SNAPSHOT_BYTES
    kernel_calls: int = 0

    def __post_init__(self) -> None:
        if (
            not 0 <= self.max_events <= 65536
            or not 8 <= self.max_tensor_bytes <= MAX_SNAPSHOT_BYTES
            or not 0 <= self.kernel_calls <= 16
            or any(index < 0 for index in self.iterations)
            or (self.kernel_calls and (self.identities or self.iterations))
        ):
            raise ValueError("trace selection exceeds its tensor/iteration/profiler bounds")


@dataclass(slots=True, eq=False)
class _TraceEntry:
    make_event: Callable[[], TraceEvent]
    # Producers pass immutable tuples or StateRefs columns; only snapshots need individual objects.
    refs: Sequence[StateRef]
    timing: tuple[torch.cuda.Event, torch.cuda.Event] | None
    payload: tuple[torch.Tensor, torch.cuda.Event | None] | None
    pending: bool
    layers: torch.Tensor | None = None

    def snapshot(self) -> TraceEvent:
        """Materialize retained immutable refs and encode a completed owned host value on inspection."""
        event = self.make_event()
        if self.payload is None and not self.refs and self.layers is None:
            return event
        payload = None
        if self.payload is not None:
            host, ready = self.payload
            if ready is not None:
                ready.synchronize()
            payload = b64encode(host.reshape(-1).view(torch.uint8).numpy().tobytes()).decode("ascii")
        if self.layers is None:
            layers = event.layers
        else:
            flat_layers = self.layers.reshape(-1)
            layers = tuple(int(flat_layers[index].item()) for index in range(flat_layers.numel()))
        return replace(event, refs=tuple(self.refs), tensor_data=payload, layers=layers)


class ExecutionTrace:
    def __init__(
        self,
        plan: ScheduleDecision,
        capacity: int = EVENT_CAPACITY,
        on_export: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        assert capacity >= 1
        self.plan = plan
        self._execution_id = uuid4().hex
        self._events: deque[_TraceEntry] = deque(maxlen=capacity)
        self._timed_tasks: frozenset[TaskId] = frozenset()
        self._selection: TraceSelection = TraceSelection()
        self._kernel_calls: int = 0
        self._sequence: int = 0
        self._dropped: int = 0
        self._lock = Condition()
        self._on_export, self._export_dropped = on_export, 0
        self._closed = False
        self._failure: BaseException | None = None
        self.thread = (
            Thread(target=self._deliver, args=(on_export,), name="pllm-export", daemon=True)
            if on_export is not None
            else None
        )
        if self.thread is not None:
            self.thread.start()
            # A daemon delivery thread still alive at interpreter exit aborts the process.
            atexit.register(self.close)

    def configure(self, selection: TraceSelection) -> None:
        if not set(selection.tasks).issubset(task.task_id for task in self.plan.tasks):
            raise ValueError("observation selection is outside the original physical plan")
        if selection.max_events and selection.max_events < (self._events.maxlen or 0):
            self.drain()
        with self._lock:
            if selection.max_events:
                self._dropped += max(0, len(self._events) - selection.max_events)
                self._events = deque(self._events, maxlen=selection.max_events)
            self._selection, self._kernel_calls = selection, selection.kernel_calls

    @contextmanager
    def private(self) -> Generator[None, None, None]:
        """Record isolated cold calls without changing the public observation log.

        The caller must own an idle execution for the whole scope. The recorder
        identity stays unchanged, including its effect namespace in Region IR.
        """
        self.drain()
        with self._lock:
            saved = (
                self._events,
                self._sequence,
                self._dropped,
                self._timed_tasks,
                self._selection,
                self._kernel_calls,
                self._on_export,
                self._export_dropped,
            )
            self._events = deque(maxlen=self._events.maxlen)
            self._sequence = self._dropped = self._kernel_calls = self._export_dropped = 0
            self._on_export = None
            self._timed_tasks, self._selection = frozenset(), TraceSelection()
        try:
            yield
        finally:
            with self._lock:
                self._collect_timings()
                (
                    self._events,
                    self._sequence,
                    self._dropped,
                    self._timed_tasks,
                    self._selection,
                    self._kernel_calls,
                    self._on_export,
                    self._export_dropped,
                ) = saved

    @property
    def observing(self) -> bool:
        return self._selection.tensors or self._selection.gradients

    def observe(
        self,
        invocation_id: int,
        task_id: TaskId,
        name: str,
        tensor: torch.Tensor,
        *,
        identities: tuple[RequestIdentity, ...] = (),
        row: PackedPhaseRow | None = None,
        refs: Sequence[StateRef] = (),
        iteration: int | None = None,
        gradient: bool = False,
        step_id: int | None = None,
    ) -> None:
        """Observe an explicit mathematical value or computed gradient, never install autograd hooks.

        Callers pass a single request's value when request isolation is required.
        Shared values are emitted only when every associated identity is selected.
        This copies values without retaining an autograd graph; CUDA capture is rejected.
        """
        selection = self._selection
        identities = (row.identity,) if row is not None else identities
        if (
            not (selection.gradients if gradient else selection.tensors)
            or selection.tasks
            and task_id not in selection.tasks
            or selection.names
            and name not in selection.names
            or selection.iterations
            and iteration not in selection.iterations
            or selection.identities
            and (not identities or not set(identities).issubset(selection.identities))
        ):
            return
        assert not (tensor.is_cuda and torch.cuda.is_current_stream_capturing())
        self.record(
            TraceEventKind.GRADIENT if gradient else TraceEventKind.TENSOR,
            invocation_id,
            task_id=task_id,
            identities=identities,
            refs=refs,
            row=row,
            iteration=iteration,
            tensor_name=name,
            tensor=self._bounded(tensor),
            step_id=step_id,
            source_shape=tuple(tensor.shape),
        )

    @property
    def records_states(self) -> bool:
        return self._selection.states

    def retired_state(self, identity: RequestIdentity, name: str, tensor: torch.Tensor) -> None:
        """Record one field of a retiring request's newest state version."""
        self.record(
            TraceEventKind.RETIRE,
            0,
            identities=(identity,),
            tensor_name=name,
            tensor=self._bounded(tensor),
            source_shape=tuple(tensor.shape),
        )

    def _bounded(self, tensor: torch.Tensor) -> torch.Tensor:
        """The selection's flat prefix of a value larger than max_tensor_bytes."""
        limit = self._selection.max_tensor_bytes
        if tensor.numel() * tensor.element_size() <= limit:
            return tensor
        return torch.take(tensor.detach(), torch.arange(limit // tensor.element_size(), device=tensor.device))

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
        if kernels and (
            stream is None
            or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities()
            or torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError("kernel profiling requires a non-capturing CUDA task and Kineto CUDA activity support")
        profiler = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    *([torch.profiler.ProfilerActivity.CUDA] if stream is not None else []),
                ]
            )
            if kernels
            else None
        )
        timing = None
        if timed and stream is not None:
            timing = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timing[0].record(stream)
        self.record(TraceEventKind.TASK_BEGIN, invocation_id, task_id=task_id)
        try:
            with profiler if profiler is not None else nullcontext():
                yield
            if timing is not None:
                timing[1].record(stream)
        except Exception as exc:
            match exc:
                case WindowBackpressure():
                    self.record(TraceEventKind.BACKPRESSURE, invocation_id, task_id=task_id)
                case _:
                    self.record(TraceEventKind.TASK_FAILED, invocation_id, task_id=task_id)
            raise
        if profiler is not None:
            events = tuple[_KernelEvent, ...](profiler.events() or ())
            for event in events:
                if event.device_type == DeviceType.CUDA:
                    self.record(
                        TraceEventKind.KERNEL,
                        invocation_id,
                        task_id=task_id,
                        kernel_name=event.name,
                        device_ordinal=event.device_index,
                        device_duration_ns=round(event.device_time_total * 1000),
                    )
        self.record(
            TraceEventKind.TASK_END,
            invocation_id,
            task_id=task_id,
            timing=timing,
            device_ordinal=None if timing is None or stream is None else stream.device.index,
        )

    def record(
        self,
        kind: TraceEventKind,
        invocation_id: int,
        *,
        task_id: TaskId | None = None,
        identities: tuple[RequestIdentity, ...] = (),
        positions: tuple[int, ...] = (),
        refs: Sequence[StateRef] = (),
        operation_id: int | None = None,
        pin_id: int | None = None,
        release_id: int | None = None,
        row: PackedPhaseRow | None = None,
        iteration: int | None = None,
        relative_l2: float | None = None,
        device_ordinal: int | None = None,
        timing: tuple[torch.cuda.Event, torch.cuda.Event] | None = None,
        tensor: torch.Tensor | None = None,
        tensor_name: str | None = None,
        step_id: int | None = None,
        source_shape: tuple[int, ...] = (),
        kernel_name: str | None = None,
        device_duration_ns: int | None = None,
        export: Literal["drop", "block"] | None = None,
        dropped_rows: int = 0,
        layers: torch.Tensor | None = None,
    ) -> None:
        if tensor is not None and tensor.is_meta:
            return
        with self._lock:
            while True:
                if export is not None and self._failure is not None:
                    raise self._failure
                assert not self._closed
                if len(self._events) < (self._events.maxlen or 0):
                    break
                evict = next((entry for entry in self._events if not entry.pending), None)
                if evict is not None:
                    self._events.remove(evict)
                    self._dropped += 1
                    break
                if export == "block":
                    self._lock.wait()
                    continue
                self._sequence += 1
                self._export_dropped += export is not None
                self._dropped += export is None
                return
            dtype, shape = None, ()
            payload = None
            host_layers = None
            if tensor is not None:
                assert tensor_name and tensor.numel() * tensor.element_size() <= MAX_SNAPSHOT_BYTES
                dtype = TensorDType(str(tensor.dtype).removeprefix("torch."))
                shape = tuple(tensor.shape)
                # Own a bounded host copy; delivery or explicit snapshot reads wait for it.
                host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=tensor.is_cuda)
                host.copy_(tensor.detach(), non_blocking=True)
                if layers is not None:
                    host_layers = torch.empty(layers.shape, dtype=torch.int64, device="cpu", pin_memory=layers.is_cuda)
                    host_layers.copy_(layers.detach(), non_blocking=True)
                ready = torch.cuda.Event() if tensor.is_cuda else None
                if ready is not None:
                    ready.record(torch.cuda.current_stream(tensor.device))
                payload = host, ready
            self._sequence += 1
            self._events.append(
                _TraceEntry(
                    partial(
                        TraceEvent,
                        self._sequence,
                        monotonic_ns(),
                        kind,
                        invocation_id,
                        task_id,
                        identities,
                        (),
                        positions,
                        (),
                        operation_id,
                        pin_id,
                        release_id,
                        row,
                        iteration,
                        relative_l2,
                        device_ordinal,
                        tensor_name=tensor_name,
                        tensor_dtype=dtype,
                        tensor_shape=shape,
                        step_id=step_id,
                        source_shape=source_shape,
                        dropped_rows=dropped_rows,
                        kernel_name=kernel_name,
                        device_duration_ns=device_duration_ns,
                    ),
                    refs,
                    timing,
                    payload,
                    export is not None and self._on_export is not None,
                    host_layers,
                )
            )

            if export is not None and self._on_export is not None:
                self._lock.notify_all()

    def _deliver(self, on_export: Callable[[TraceEvent], None]) -> None:
        while True:
            with self._lock:
                self._lock.wait_for(lambda: self._closed or any(entry.pending for entry in self._events))
                entry = next((entry for entry in self._events if entry.pending), None)
                if entry is None:
                    return
            try:
                on_export(entry.snapshot())
            except BaseException as exc:
                with self._lock:
                    self._failure = exc
                    self._lock.notify_all()
                return
            with self._lock:
                entry.pending = False
                self._lock.notify_all()

    def drain(self) -> None:
        """Wait through the current sequence, and surface the first consumer failure."""
        with self._lock:
            watermark = self._sequence
            self._lock.wait_for(
                lambda: (
                    self._failure is not None
                    or not any(entry.pending and entry.make_event().sequence <= watermark for entry in self._events)
                )
            )
            if self._failure is not None:
                raise self._failure

    def _collect_timings(self) -> None:
        """Finish history's outstanding host copies before releasing their storage."""
        for entry in self._events:
            if entry.payload is not None and entry.payload[1] is not None:
                entry.payload[1].synchronize()

    def close(self) -> None:
        """Stop admission, finish copies and delivery, and release the bounded history; idempotent."""
        if self._closed:
            return
        atexit.unregister(self.close)
        with self._lock:
            self._closed = True
            self._lock.notify_all()
        try:
            self.drain()
        finally:
            if self.thread is not None:
                self.thread.join()
            with self._lock:
                self._collect_timings()
                self._events.clear()

    def snapshot(self) -> TraceSnapshot:
        with self._lock:
            events: list[TraceEvent] = []
            pending = 0
            for entry in self._events:
                event, timing = entry.snapshot(), entry.timing
                if timing is not None:
                    if timing[1].query():
                        event = replace(event, device_duration_ns=round(timing[0].elapsed_time(timing[1]) * 1_000_000))
                    else:
                        pending += 1
                events.append(event)
            return TraceSnapshot(
                self._execution_id,
                os.getpid(),
                tuple(events),
                self._dropped,
                self._events.maxlen or 0,
                pending,
                self._export_dropped,
            )
