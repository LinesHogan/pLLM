"""Safe-point publication and retirement for physical WorkGraph plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import TypeVar, cast

import torch

from tllm.runtime.scheduling.physical import (
    BufferLease,
    EventLike,
    Fence,
    GraphLike,
    StreamLike,
)
from tllm.runtime.scheduling.phase3_transfer import (
    CopyHandle,
    PlacementImplementationBinding,
    PlacementPlanDispatcher,
    TensorLike,
)
from tllm.runtime.scheduling.phase5_adaptive import (
    AdaptiveFlowBinding,
    AdaptiveTransferStatus,
    Phase5AdaptiveScheduler,
    validate_adaptive_flow_bindings,
)
from tllm.workgraph.frontend import ArtifactKey, Device
from tllm.workgraph.physical import (
    BufferBucketDesc,
    CapturePolicy,
    FenceKey,
    PhysicalFlowPlan,
    StreamClass,
)
from tllm.workgraph.placement import PlacementPlan
from tllm.workgraph.admission import CandidateActionDesc, AdmissionControlPlan


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
    retire_fence: Fence

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.key), StreamClass):
            raise TypeError("RuntimeStream.key must be StreamClass")
        stream = cast(object, self.stream)
        if not isinstance(stream, StreamLike):
            raise TypeError("RuntimeStream.stream must implement wait_event(event)")
        if not isinstance(cast(object, self.retire_fence), Fence):
            raise TypeError("RuntimeStream.retire_fence must be Fence")
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
    placement_plan: PlacementPlan | None = None
    placement_dispatcher: PlacementPlanDispatcher | None = None
    placement_implementations: tuple[PlacementImplementationBinding, ...] = ()
    adaptive_plan: AdmissionControlPlan | None = None
    adaptive_scheduler: Phase5AdaptiveScheduler | None = None
    adaptive_flow_bindings: tuple[AdaptiveFlowBinding, ...] = ()
    leases: tuple[BufferLease, ...] = ()
    streams: tuple[RuntimeStream, ...] = ()
    events: tuple[RuntimeEvent, ...] = ()
    graphs: tuple[RuntimeGraph, ...] = ()
    _adaptive_transfers: dict[
        tuple[str, object, int],
        tuple[CandidateActionDesc, tuple[CopyHandle, ...]],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _deferred_handoffs: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        raw_plan = cast(object, self.plan)
        if not isinstance(raw_plan, PhysicalFlowPlan):
            raise TypeError("PlanResources.plan must be PhysicalFlowPlan")
        placement_plan = self.placement_plan
        placement_dispatcher = self.placement_dispatcher
        placement_implementations = _typed_tuple(
            self.placement_implementations,
            PlacementImplementationBinding,
            "PlanResources.placement_implementations",
        )
        if (placement_plan is None) != (placement_dispatcher is None):
            raise ValueError(
                "PlanResources placement plan and dispatcher must be published together"
            )
        if placement_plan is not None:
            if not isinstance(cast(object, placement_plan), PlacementPlan):
                raise TypeError("PlanResources.placement_plan must be PlacementPlan")
            if not isinstance(
                cast(object, placement_dispatcher), PlacementPlanDispatcher
            ):
                raise TypeError(
                    "PlanResources.placement_dispatcher must be PlacementPlanDispatcher"
                )
            assert placement_dispatcher is not None
            if placement_plan.plan_epoch != raw_plan.plan_epoch:
                raise ValueError(
                    "PlanResources physical and placement epochs must match"
                )
            if placement_dispatcher.plan != placement_plan:
                raise ValueError(
                    "PlanResources dispatcher must bind its frozen placement plan"
                )
            physical_executors = {
                node.node_key: node.executor for node in raw_plan.nodes
            }
            placement_executors = {
                selection.node_key: selection.logical_executor
                for selection in placement_plan.selections
            }
            if placement_executors != physical_executors:
                raise ValueError(
                    "PlanResources placement selections must match physical nodes and executors"
                )
            expected_implementations = {
                (selection.node_key, selection.key)
                for selection in placement_plan.selections
            }
            actual_implementations = {
                (binding.node_key, binding.implementation_key)
                for binding in placement_implementations
            }
            if (
                len(actual_implementations) != len(placement_implementations)
                or actual_implementations != expected_implementations
            ):
                raise ValueError(
                    "PlanResources implementation bindings must exactly cover placement selections"
                )
        elif placement_implementations:
            raise ValueError(
                "PlanResources implementation bindings require a placement plan"
            )
        adaptive_plan = self.adaptive_plan
        adaptive_scheduler = self.adaptive_scheduler
        adaptive_flow_bindings = _typed_tuple(
            self.adaptive_flow_bindings,
            AdaptiveFlowBinding,
            "PlanResources.adaptive_flow_bindings",
        )
        if (adaptive_plan is None) != (adaptive_scheduler is None):
            raise ValueError(
                "PlanResources adaptive plan and scheduler must be published together"
            )
        if adaptive_plan is not None:
            if not isinstance(cast(object, adaptive_plan), AdmissionControlPlan):
                raise TypeError("PlanResources.adaptive_plan must be AdmissionControlPlan")
            if not isinstance(
                cast(object, adaptive_scheduler), Phase5AdaptiveScheduler
            ):
                raise TypeError(
                    "PlanResources.adaptive_scheduler must be Phase5AdaptiveScheduler"
                )
            if placement_plan is None or adaptive_plan.placement_plan != placement_plan:
                raise ValueError(
                    "PlanResources adaptive plan must bind its exact placement plan"
                )
            if adaptive_plan.plan_epoch != raw_plan.plan_epoch:
                raise ValueError(
                    "PlanResources physical and adaptive epochs must match"
                )
            assert adaptive_scheduler is not None
            prepared = adaptive_scheduler.prepared
            if prepared is None or prepared.plan != adaptive_plan:
                raise ValueError(
                    "PlanResources adaptive scheduler must prepare its frozen plan"
                )
            flow_ids = tuple(binding.flow_id for binding in adaptive_flow_bindings)
            if len(set(flow_ids)) != len(flow_ids):
                raise ValueError(
                    "PlanResources adaptive flow bindings must use unique flow ids"
                )
            adaptive_flow_bindings = validate_adaptive_flow_bindings(
                adaptive_plan,
                adaptive_flow_bindings,
            )
        elif adaptive_flow_bindings:
            raise ValueError(
                "PlanResources adaptive flow bindings require an adaptive plan"
            )
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
        retire_keys = tuple(resource.retire_fence.key for resource in streams)
        if len(set(retire_keys)) != len(retire_keys):
            raise ValueError("PlanResources contains duplicate retire fence keys")
        externally_bound_fences = {
            resource.key: resource.event for resource in events
        }
        externally_bound_fences.update(lease_events)
        if any(
            resource.retire_fence.key in externally_bound_fences
            and externally_bound_fences[resource.retire_fence.key]
            is not resource.retire_fence.event
            for resource in streams
        ):
            raise ValueError(
                "PlanResources binds one fence key to different event objects"
            )
        object.__setattr__(self, "plan", raw_plan)
        object.__setattr__(self, "placement_plan", placement_plan)
        object.__setattr__(self, "placement_dispatcher", placement_dispatcher)
        object.__setattr__(
            self,
            "placement_implementations",
            placement_implementations,
        )
        object.__setattr__(self, "adaptive_plan", adaptive_plan)
        object.__setattr__(self, "adaptive_scheduler", adaptive_scheduler)
        object.__setattr__(
            self,
            "adaptive_flow_bindings",
            adaptive_flow_bindings,
        )
        object.__setattr__(self, "leases", leases)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "graphs", graphs)

    @property
    def reusable(self) -> bool:
        return (
            all(lease.reusable for lease in self.leases)
            and all(
                resource.retire_fence.recorded and resource.retire_fence.complete
                for resource in self.streams
            )
            and (
                self.placement_dispatcher is None
                or self.placement_dispatcher.in_flight == 0
            )
            and not self._adaptive_transfers
            and self._deferred_handoffs == 0
        )

    def retain_deferred_handoff(self) -> None:
        object.__setattr__(self, "_deferred_handoffs", self._deferred_handoffs + 1)

    def release_deferred_handoff(self) -> None:
        if self._deferred_handoffs <= 0:
            raise RuntimeError("adaptive deferred handoff ownership is unbalanced")
        object.__setattr__(self, "_deferred_handoffs", self._deferred_handoffs - 1)

    def poll_transfers(self) -> None:
        if self.placement_dispatcher is not None:
            self.placement_dispatcher.poll()

    def adaptive_flow_binding(self, flow_id: str) -> AdaptiveFlowBinding | None:
        if not isinstance(flow_id, str) or not flow_id:
            raise ValueError("adaptive flow lookup requires a non-empty flow id")
        return next(
            (
                binding
                for binding in self.adaptive_flow_bindings
                if binding.flow_id == flow_id
            ),
            None,
        )

    @staticmethod
    def _adaptive_transfer_token(
        binding: AdaptiveFlowBinding,
        bundle_key: object,
        bundle_version: int,
    ) -> tuple[str, object, int]:
        try:
            hash(bundle_key)
        except TypeError as exc:
            raise TypeError("adaptive transfer bundle key must be hashable") from exc
        if not isinstance(bundle_version, int) or isinstance(bundle_version, bool):
            raise TypeError("adaptive transfer bundle version must be int")
        return binding.flow_id, bundle_key, bundle_version

    def adaptive_pending_action(
        self,
        binding: AdaptiveFlowBinding,
        *,
        bundle_key: object,
        bundle_version: int,
    ) -> CandidateActionDesc | None:
        token = self._adaptive_transfer_token(
            binding,
            bundle_key,
            bundle_version,
        )
        batch = self._adaptive_transfers.get(token)
        return None if batch is None else batch[0]

    def abandon_adaptive_transfer(
        self,
        *,
        flow_id: str,
        bundle_key: object,
        bundle_version: int,
    ) -> None:
        """Drop a discarded callback's token while its copy fence stays owned."""

        binding = self.adaptive_flow_binding(flow_id)
        if binding is None:
            return
        token = self._adaptive_transfer_token(
            binding,
            bundle_key,
            bundle_version,
        )
        self._adaptive_transfers.pop(token, None)

    def advance_adaptive_transfers(
        self,
        binding: AdaptiveFlowBinding,
        *,
        bundle_key: object,
        bundle_version: int,
        entries: Mapping[str, object],
        candidate: CandidateActionDesc,
        must_resolve: bool,
    ) -> AdaptiveTransferStatus:
        """Start or poll the exact Phase 3 MoveSet selected for one bundle."""

        if not isinstance(binding, AdaptiveFlowBinding):
            raise TypeError("adaptive transfer requires AdaptiveFlowBinding")
        if not isinstance(candidate, CandidateActionDesc):
            raise TypeError("adaptive transfer requires CandidateActionDesc")
        if not isinstance(entries, Mapping):
            raise TypeError("adaptive transfer entries must be a mapping")
        move_set_key = candidate.move_set
        if move_set_key is None:
            return AdaptiveTransferStatus.READY
        adaptive_plan = self.adaptive_plan
        dispatcher = self.placement_dispatcher
        if adaptive_plan is None or dispatcher is None:
            raise RuntimeError("adaptive transfer requires a prepared placement epoch")
        move_set = next(
            item for item in adaptive_plan.move_sets if item.key == move_set_key
        )
        if not move_set.moves:
            return AdaptiveTransferStatus.READY
        transfer = binding.transfer_for(move_set_key)
        if transfer is None:
            raise RuntimeError("adaptive MoveSet lacks its explicit transfer binding")
        token = self._adaptive_transfer_token(
            binding,
            bundle_key,
            bundle_version,
        )
        batch = self._adaptive_transfers.get(token)
        if batch is None:
            try:
                source = entries[transfer.source_entry]
                destination = entries[transfer.destination_entry]
            except KeyError as exc:
                raise KeyError(
                    "adaptive transfer bundle lacks an explicitly bound payload"
                ) from exc
            handle = dispatcher.submit_scheduled(
                move_set.moves,
                source=cast(TensorLike, source),
                destination=cast(TensorLike, destination),
            )
            batch = candidate, (handle,)
            self._adaptive_transfers[token] = batch
        elif batch[0] != candidate:
            raise RuntimeError("adaptive bundle changed action while transfers were pending")

        dispatcher.poll()
        handles = batch[1]
        failed = next((handle.error for handle in handles if handle.error is not None), None)
        if failed is not None:
            self._adaptive_transfers.pop(token, None)
            raise RuntimeError("adaptive Phase 3 transfer failed") from failed
        if all(handle.complete for handle in handles):
            self._adaptive_transfers.pop(token, None)
            return AdaptiveTransferStatus.READY
        if must_resolve:
            raise RuntimeError("adaptive Phase 3 transfer missed the bundle deadline")
        return AdaptiveTransferStatus.PENDING

    def reset_adaptive_reservations(self) -> None:
        if self.adaptive_scheduler is not None:
            self.adaptive_scheduler.reset_reservations()


class PlanSnapshotStore:
    """Atomically publish plan epochs at prepare safe points.

    An active step retains the exact :class:`PlanResources` object it acquired,
    even when a later step publishes a staged epoch.
    """

    __slots__ = (
        "_current",
        "_pending",
        "_active",
        "_retired",
        "_acquired",
        "_lock",
    )

    def __init__(self, initial: PlanResources) -> None:
        if not isinstance(cast(object, initial), PlanResources):
            raise TypeError("PlanSnapshotStore.initial must be PlanResources")
        self._current = initial
        self._pending: PlanResources | None = None
        self._active: dict[int, PlanResources] = {}
        self._retired: list[PlanResources] = []
        self._acquired: set[int] = set()
        self._lock = Lock()

    @property
    def current(self) -> PlanResources:
        with self._lock:
            return self._current

    @property
    def retired_count(self) -> int:
        with self._lock:
            return len(self._retired)

    @property
    def closeable(self) -> bool:
        """Return whether every acquired epoch can be released without waiting."""

        with self._lock:
            if self._active:
                return False
            resources = (self._current, *self._retired)
            return all(
                id(item) not in self._acquired or item.reusable
                for item in resources
            )

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
            self._acquired.add(id(self._current))
            return self._current

    def active(self, step_id: int) -> PlanResources:
        with self._lock:
            return self._active[_step_id(step_id)]

    def adaptive_transfer_owner(
        self,
        *,
        flow_id: str,
        bundle_key: object,
        bundle_version: int,
    ) -> PlanResources | None:
        """Find the retained epoch that owns an in-flight adaptive transfer."""

        with self._lock:
            candidates = (
                *self._active.values(),
                self._current,
                *self._retired,
            )
        seen: set[int] = set()
        for resources in candidates:
            resource_id = id(resources)
            if resource_id in seen:
                continue
            seen.add(resource_id)
            binding = resources.adaptive_flow_binding(flow_id)
            if binding is not None and resources.adaptive_pending_action(
                binding,
                bundle_key=bundle_key,
                bundle_version=bundle_version,
            ) is not None:
                return resources
        return None

    def finish_step(self, step_id: int) -> None:
        with self._lock:
            self._active.pop(_step_id(step_id), None)

    def collect_retired(self) -> int:
        with self._lock:
            active_ids = {id(resources) for resources in self._active.values()}
            kept: list[PlanResources] = []
            collected = 0
            for resources in self._retired:
                resources.poll_transfers()
                resource_id = id(resources)
                if resource_id in active_ids or (
                    resource_id in self._acquired and not resources.reusable
                ):
                    kept.append(resources)
                else:
                    collected += 1
                    self._acquired.discard(resource_id)
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
        not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0
        for dimension in shape
    ):
        raise TypeError("select_buffer_bucket.shape must contain non-negative ints")
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
