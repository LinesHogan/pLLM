"""Safe-point publication and retirement for physical WorkGraph plans."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import TypeVar, cast

import torch

from tllm.runtime.scheduling.physical import (
    BufferLease,
    EventLike,
    GraphLike,
    StreamLike,
)
from tllm.workgraph.frontend import ArtifactKey, Device
from tllm.workgraph.physical import (
    BufferBucketDesc,
    CapturePolicy,
    FenceKey,
    PhysicalFlowPlan,
    StreamClass,
)


_T = TypeVar("_T")


def _typed_tuple(value: object, expected: type[_T], label: str) -> tuple[_T, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be tuple")
    items = cast(tuple[object, ...], value)
    if any(not isinstance(item, expected) for item in items):
        raise TypeError(f"{label} must contain only {expected.__name__}")
    return cast(tuple[_T, ...], items)


@dataclass(frozen=True, slots=True)
class RuntimeStream:
    """Python-owned stream associated with one frozen stream class."""

    key: StreamClass
    stream: object

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.key), StreamClass):
            raise TypeError("RuntimeStream.key must be StreamClass")
        stream = cast(object, self.stream)
        if not isinstance(stream, StreamLike):
            raise TypeError("RuntimeStream.stream must implement wait_event(event)")
        object.__setattr__(self, "stream", stream)


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Python-owned event associated with one frozen fence identity."""

    key: FenceKey
    event: object

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.key), FenceKey):
            raise TypeError("RuntimeEvent.key must be FenceKey")
        event = cast(object, self.event)
        if not isinstance(event, EventLike):
            raise TypeError("RuntimeEvent.event must implement record(stream) and query()")
        object.__setattr__(self, "event", event)


@dataclass(frozen=True, slots=True)
class RuntimeGraph:
    """Python-owned replay graph associated with one capture artifact."""

    key: ArtifactKey
    graph: object

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.key), ArtifactKey):
            raise TypeError("RuntimeGraph.key must be ArtifactKey")
        if len(self.key.digest) != 64:
            raise ValueError("RuntimeGraph.key requires a SHA-256 artifact digest")
        graph = cast(object, self.graph)
        if not isinstance(graph, GraphLike):
            raise TypeError("RuntimeGraph.graph must implement replay()")
        object.__setattr__(self, "graph", graph)


@dataclass(frozen=True, slots=True, eq=False)
class PlanResources:
    """One physical plan and the Python runtime resources owned by its epoch."""

    plan: PhysicalFlowPlan
    leases: tuple[BufferLease, ...] = ()
    streams: tuple[RuntimeStream, ...] = ()
    events: tuple[RuntimeEvent, ...] = ()
    graphs: tuple[RuntimeGraph, ...] = ()

    def __post_init__(self) -> None:
        raw_plan = cast(object, self.plan)
        if not isinstance(raw_plan, PhysicalFlowPlan):
            raise TypeError("PlanResources.plan must be PhysicalFlowPlan")
        leases = _typed_tuple(self.leases, BufferLease, "PlanResources.leases")
        streams = _typed_tuple(self.streams, RuntimeStream, "PlanResources.streams")
        events = _typed_tuple(self.events, RuntimeEvent, "PlanResources.events")
        graphs = _typed_tuple(self.graphs, RuntimeGraph, "PlanResources.graphs")
        if any(lease.plan_epoch != raw_plan.plan_epoch for lease in leases):
            raise ValueError("PlanResources leases must belong to the plan epoch")
        bucket_keys = {bucket.key for bucket in raw_plan.buffer_buckets}
        if any(lease.bucket.key not in bucket_keys for lease in leases):
            raise ValueError("PlanResources lease references an unknown plan bucket")
        stream_keys = {node.stream for node in raw_plan.nodes}
        if any(resource.key not in stream_keys for resource in streams):
            raise ValueError("PlanResources stream is not declared by its physical plan")
        graph_keys = {
            node.artifact_key
            for node in raw_plan.nodes
            if node.capture_policy != CapturePolicy.DISABLED
            and node.artifact_key is not None
        }
        if any(resource.key not in graph_keys for resource in graphs):
            raise ValueError("PlanResources graph is not enabled by its physical plan")
        event_keys = {
            dependency.fence
            for node in raw_plan.nodes
            for dependency in node.fences
        }
        lease_fences = tuple(
            fence
            for lease in leases
            for fence in (
                lease.ready_fence,
                lease.completion_fence,
                lease.retire_fence,
            )
            if fence is not None
        )
        event_keys.update(fence.key for fence in lease_fences)
        if any(resource.key not in event_keys for resource in events):
            raise ValueError("PlanResources event is not declared by its plan or leases")
        lease_events: dict[FenceKey, object] = {}
        for fence in lease_fences:
            existing_event = lease_events.get(fence.key)
            if existing_event is not None and existing_event is not fence.event:
                raise ValueError(
                    "PlanResources leases bind one fence key to different event objects"
                )
            lease_events[fence.key] = fence.event
        if any(
            resource.key in lease_events
            and resource.event is not lease_events[resource.key]
            for resource in events
        ):
            raise ValueError("PlanResources lease event binding uses a different event object")
        for label, keys in (
            ("stream", tuple(resource.key for resource in streams)),
            ("event", tuple(resource.key for resource in events)),
            ("graph", tuple(resource.key for resource in graphs)),
        ):
            if len(set(keys)) != len(keys):
                raise ValueError(f"PlanResources contains duplicate {label} resource keys")
        object.__setattr__(self, "plan", raw_plan)
        object.__setattr__(self, "leases", leases)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "graphs", graphs)


class PlanSnapshotStore:
    """Atomically publish plan epochs at prepare safe points.

    An active step retains the exact :class:`PlanResources` object it acquired,
    even when a later step publishes a staged epoch.
    """

    __slots__ = ("_current", "_pending", "_active", "_retired", "_lock")

    def __init__(self, initial: PlanResources) -> None:
        if not isinstance(cast(object, initial), PlanResources):
            raise TypeError("PlanSnapshotStore.initial must be PlanResources")
        self._current = initial
        self._pending: PlanResources | None = None
        self._active: dict[int, PlanResources] = {}
        self._retired: list[PlanResources] = []
        self._lock = Lock()

    @property
    def current(self) -> PlanResources:
        with self._lock:
            return self._current

    @property
    def retired_count(self) -> int:
        with self._lock:
            return len(self._retired)

    def stage(self, resources: PlanResources) -> None:
        if not isinstance(cast(object, resources), PlanResources):
            raise TypeError("PlanSnapshotStore.stage resources must be PlanResources")
        with self._lock:
            floor = self._current.plan.plan_epoch
            if self._pending is not None:
                floor = max(floor, self._pending.plan.plan_epoch)
            if resources.plan.plan_epoch <= floor:
                raise ValueError("staged physical plan epoch must be strictly newer")
            self._pending = resources

    def prepare_step(self, step_id: int) -> PlanResources:
        step = _step_id(step_id)
        with self._lock:
            existing = self._active.get(step)
            if existing is not None:
                return existing
            pending = self._pending
            if pending is not None:
                previous = self._current
                self._current = pending
                self._pending = None
                if all(resources is not previous for resources in self._retired):
                    self._retired.append(previous)
            self._active[step] = self._current
            return self._current

    def active(self, step_id: int) -> PlanResources:
        with self._lock:
            return self._active[_step_id(step_id)]

    def finish_step(self, step_id: int) -> None:
        with self._lock:
            self._active.pop(_step_id(step_id), None)

    def collect_retired(self) -> int:
        with self._lock:
            active_ids = {id(resources) for resources in self._active.values()}
            kept: list[PlanResources] = []
            collected = 0
            for resources in self._retired:
                if id(resources) in active_ids or any(
                    not lease.reusable for lease in resources.leases
                ):
                    kept.append(resources)
                else:
                    collected += 1
            self._retired = kept
            return collected


def _step_id(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("step_id must be int")
    if value < 0:
        raise ValueError("step_id must be non-negative")
    return value


def select_buffer_bucket(
    plan: PhysicalFlowPlan,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> BufferBucketDesc | None:
    """Return an exact specialization bucket, or ``None`` for eager fallback."""

    if not isinstance(cast(object, plan), PhysicalFlowPlan):
        raise TypeError("select_buffer_bucket.plan must be PhysicalFlowPlan")
    if not isinstance(cast(object, shape), tuple) or any(
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1
        for dimension in shape
    ):
        raise TypeError("select_buffer_bucket.shape must contain positive ints")
    if not isinstance(cast(object, dtype), torch.dtype):
        raise TypeError("select_buffer_bucket.dtype must be torch.dtype")
    if not isinstance(cast(object, device), torch.device):
        raise TypeError("select_buffer_bucket.device must be torch.device")
    dtype_name = _DTYPE_NAMES.get(dtype)
    if dtype_name is None:
        return None
    for bucket in plan.buffer_buckets:
        if bucket.shape != shape:
            continue
        if bucket.dtype.namespace != "torch" or bucket.dtype.name != dtype_name:
            continue
        if bucket.device.kind.value != device.type:
            continue
        if bucket.device.kind == Device.CUDA and bucket.device.index != device.index:
            continue
        return bucket
    return None


_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.bfloat16: "bfloat16",
    torch.bool: "bool",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
}


__all__ = [
    "PlanResources",
    "PlanSnapshotStore",
    "RuntimeEvent",
    "RuntimeGraph",
    "RuntimeStream",
    "select_buffer_bucket",
]
