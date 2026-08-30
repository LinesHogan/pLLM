"""Runtime binding for compiler-owned auxiliary workload plans.

This module intentionally admits only workloads whose complete physical plan
has a fixed executor binding. Unsupported declarations remain on their
migration adapter; a supported plan with a missing runtime executor fails
before the first model forward.
"""

from __future__ import annotations

import contextlib
from collections import Counter, deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import torch

from tllm.contracts.request_identity import RequestFinishedBatch
from tllm.workload.staging import PortTransactionStatus

WorkloadInstanceT = TypeVar("WorkloadInstanceT")


class WorkloadPortOwner(Protocol):
    def owns_port(self, key: str) -> bool: ...

    def prepare_port(self, key: str, transaction: str, value: object) -> None: ...

    def commit_port(self, key: str, transaction: str) -> int: ...

    def abort_port(self, key: str, transaction: str) -> None: ...

    def port_transaction_status(
        self, key: str, transaction: str
    ) -> PortTransactionStatus: ...

    def acknowledge_remote_port_consumed(
        self, key: str, transaction: str
    ) -> None: ...

    def owns_output(self, key: str) -> bool: ...

    def drain_output(self, key: str) -> tuple[object, ...]: ...

    def stage_request_finished(self, batch: RequestFinishedBatch) -> int: ...

    def stage_dense_operand_captured(
        self, batch: DenseOperandCaptureBatch
    ) -> int: ...

from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from tllm.contracts.shape_specialization import (
    RegionShapeSpecializationInput,
    RegionShapeSpecializationRequest,
)
from tllm.runtime.region_cost_probe import RegionCostProbe, RegionCostReport
from tllm.runtime.parameter_overlay import (
    DenseOperandCaptureBatch,
    OverlayChannel,
    OverlayVersionStore,
)
from tllm.runtime.workload_bindings import (
    NoPerRequestStats,
    NoSamplerBinding,
    PerRequestStats,
    PerRequestStatsService,
    RuntimePortRegistry,
    RuntimeInputBinding,
    RuntimeOutputBinding,
    SamplerBinding,
    SamplerRuntimeBinding,
)
from tllm.runtime.scheduling.physical import Fence
from tllm.runtime.scheduling.plan_store import (
    PlanResources,
    PlanSnapshotStore,
    RuntimeEvent,
    RuntimeGraph,
    RuntimeStream,
)
from tllm.runtime.scheduling.workgraph_executor import (
    PhysicalExecutionContext,
    PhysicalExecutor,
    ShapeSpecializationMiss,
    runtime_stats_from_executor,
)
from tllm.contracts.physical_plan import (
    ArtifactKey,
    CapturePolicy,
    ExecutorKey,
    FallbackPolicy,
    FenceKey,
    NodeKey,
    PhysicalNodePlan,
    StreamClass,
)
from tllm.workload.region_compiler import compare_region_outputs
from tllm.workload.runtime_values import SKIPPED
from tllm.schedule.contracts import (
    BackendObservableEvent,
    BackendRegionBinderStage,
    ExecutionRegion,
    HiddenLayerKind,
    RegionInputBinding,
    RegionInputKind,
    RegionOutputBinding,
    RegionOutputKind,
    ScheduleEffect,
    ScheduleEffectKind,
    ScheduleGraph,
    ScheduleDecision,
    ScheduleState,
)

if TYPE_CHECKING:
    from tllm.ports.residual_stream import ResidualLocator
    from tllm.contracts.consumer_stats import TrainingStats
    from tllm.contracts.runtime_stats import WorkGraphRuntimeStats
    from tllm.runtime.backend_region_binders import (
        BackendRegionBinderSlot,
    )
    from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider
    from tllm.runtime.state import ResidualRuntimeState
    from tllm.runtime.workload_plugins import StaticBackendBindingEntry
    from tllm.schedule import WorkloadCompilation
    from tllm.workload.frontend import WorkloadDeclaration
    from tllm.workload.region_compiler import RegionCallableArtifact




@dataclass(frozen=True, slots=True)
class _ExecutableMaterializationBinding:
    """One immutable, signed cross-Region value boundary.

    The planner emits one ``MaterializedValue`` per consumer Region.  This
    record carries that planner identity into the local executor without
    allowing artifact bindings to create or merge boundaries.
    """

    materialization_id: str
    value_id: str
    version: int
    producer_region: str
    consumer_region: str
    size_bytes: int
    live_start: str
    live_end: str


class _SignedMaterializationBindings:
    """Read-only executable materialization table for one signed decision."""

    __slots__ = (
        "_by_id",
        "_inputs",
        "_outputs",
    )

    def __init__(
        self,
        bindings: tuple[_ExecutableMaterializationBinding, ...],
    ) -> None:
        by_id = {binding.materialization_id: binding for binding in bindings}
        if len(by_id) != len(bindings):
            raise RuntimeError("signed materialization table contains duplicate ids")
        inputs = {
            (binding.consumer_region, binding.materialization_id): binding
            for binding in bindings
        }
        outputs = {
            (binding.producer_region, binding.materialization_id): binding
            for binding in bindings
        }
        self._by_id = MappingProxyType(by_id)
        self._inputs = MappingProxyType(inputs)
        self._outputs = MappingProxyType(outputs)
    def input_binding_for(
        self, region_id: str, materialization_id: str
    ) -> _ExecutableMaterializationBinding:
        try:
            return self._inputs[(region_id, materialization_id)]
        except KeyError as exc:
            raise RuntimeError(
                "local Region artifact has an unsigned materialized input "
                f"{materialization_id!r} in {region_id!r}"
            ) from exc

    def output_binding_for(
        self, region_id: str, materialization_id: str
    ) -> _ExecutableMaterializationBinding:
        try:
            return self._outputs[(region_id, materialization_id)]
        except KeyError as exc:
            raise RuntimeError(
                "local Region artifact has an unsigned materialized output "
                f"{materialization_id!r} in {region_id!r}"
            ) from exc


def lower_signed_materialization_bindings(
    graph: ScheduleGraph,
    decision: ScheduleDecision,
    artifacts: tuple["RegionCallableArtifact", ...],
) -> _SignedMaterializationBindings:
    """Lower Rust-signed boundary records into the only local value table.

    Artifact bindings are local callable plumbing.  They are checked against
    the signed Region and materialization records, but cannot define a value
    lifetime, producer, or consumer count themselves.
    """

    if decision.schema_version != 2:
        raise TypeError("decision must be a ScheduleDecision schema v2")

    regions_by_id = {region.region_id: region for region in decision.execution_regions}
    artifacts_by_id: dict[str, RegionCallableArtifact] = {}
    for artifact in artifacts:
        if artifact.region_id in artifacts_by_id:
            raise RuntimeError(
                f"duplicate local Region artifact {artifact.region_id!r}"
            )
        artifacts_by_id[artifact.region_id] = artifact
    missing_artifacts = sorted(set(regions_by_id) - set(artifacts_by_id))
    extra_artifacts = sorted(set(artifacts_by_id) - set(regions_by_id))
    if missing_artifacts or extra_artifacts:
        raise RuntimeError(
            "local Region artifact coverage mismatch for materialization lowering: "
            f"missing={missing_artifacts}, unexpected={extra_artifacts}"
        )

    signed_inputs: dict[tuple[str, str], RegionInputBinding] = {}
    signed_outputs: dict[tuple[str, str], RegionOutputBinding] = {}
    artifact_inputs: dict[tuple[str, str], RegionInputBinding] = {}
    artifact_outputs: dict[tuple[str, str], RegionOutputBinding] = {}
    for region_id, region in regions_by_id.items():
        artifact = artifacts_by_id[region_id]
        for binding in region.input_bindings:
            if binding.kind is RegionInputKind.MATERIALIZED:
                signed_inputs[(region_id, binding.binding_id)] = binding
        for binding in region.output_bindings:
            if binding.kind is RegionOutputKind.MATERIALIZED:
                signed_outputs[(region_id, binding.binding_id)] = binding
        for binding in artifact.input_bindings:
            if binding.kind is RegionInputKind.MATERIALIZED:
                artifact_inputs[(region_id, binding.binding_id)] = binding
        for binding in artifact.output_bindings:
            if binding.kind is RegionOutputKind.MATERIALIZED:
                artifact_outputs[(region_id, binding.binding_id)] = binding

    if artifact_inputs != signed_inputs:
        raise RuntimeError("local Region artifact materialized input bindings differ from the signed decision")
    if artifact_outputs != signed_outputs:
        raise RuntimeError("local Region artifact materialized output bindings differ from the signed decision")
    for region_id, region in regions_by_id.items():
        artifact = artifacts_by_id[region_id]
        if artifact.input_bindings != region.input_bindings:
            raise RuntimeError(
                f"local Region artifact {region_id!r} input bindings differ from the signed decision"
            )
        if artifact.output_bindings != region.output_bindings:
            raise RuntimeError(
                f"local Region artifact {region_id!r} output bindings differ from the signed decision"
            )

    values = {(value.value_id, value.version): value for value in graph.values}
    bindings: list[_ExecutableMaterializationBinding] = []
    expected_inputs: set[tuple[str, str]] = set()
    expected_outputs: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()
    for materialization in decision.materializations:
        if materialization.materialization_id in seen_ids:
            raise RuntimeError("signed materialization table contains duplicate ids")
        seen_ids.add(materialization.materialization_id)
        producer = regions_by_id.get(materialization.producer_region)
        consumer = regions_by_id.get(materialization.consumer_region)
        if producer is None or consumer is None:
            raise RuntimeError("signed materialization references an unknown Region")
        if producer.region_id == consumer.region_id:
            raise RuntimeError("Region-internal values cannot enter the materialization table")
        if (
            materialization.live_start != producer.placement_anchor
            or materialization.live_end != consumer.placement_anchor
        ):
            raise RuntimeError("signed materialization lifetime does not match Region anchors")
        value = values.get((materialization.value_id, materialization.version))
        if value is None:
            raise RuntimeError("signed materialization value or version is not present in the graph")
        if not value.producer_output:
            raise RuntimeError(
                "signed materialization value has no executable producer output"
            )
        input_key = (consumer.region_id, materialization.materialization_id)
        output_key = (producer.region_id, materialization.materialization_id)
        input_binding = signed_inputs.get(input_key)
        output_binding = signed_outputs.get(output_key)
        if input_binding is None:
            raise RuntimeError("signed materialization is missing its Region materialized input binding")
        if output_binding is None:
            raise RuntimeError("signed materialization is missing its Region materialized output binding")
        if input_binding.source != materialization.value_id:
            raise RuntimeError("signed materialization input binding has the wrong value source")
        if output_binding.target != materialization.value_id:
            raise RuntimeError("signed materialization output binding has the wrong value target")
        if (
            output_binding.producer.task_id != value.producer
            or output_binding.producer.port_name != value.producer_output
        ):
            raise RuntimeError("signed materialization output binding has the wrong producer")
        expected_inputs.add(input_key)
        expected_outputs.add(output_key)
        bindings.append(
            _ExecutableMaterializationBinding(
                materialization_id=materialization.materialization_id,
                value_id=materialization.value_id,
                version=materialization.version,
                producer_region=producer.region_id,
                consumer_region=consumer.region_id,
                size_bytes=materialization.size_bytes,
                live_start=materialization.live_start,
                live_end=materialization.live_end,
            )
        )
    if set(signed_inputs) != expected_inputs:
        raise RuntimeError("signed Region has materialized input bindings without a materialization record")
    if set(signed_outputs) != expected_outputs:
        raise RuntimeError("signed Region has materialized output bindings without a materialization record")
    return _SignedMaterializationBindings(tuple(bindings))


class _NativeShapeSpecializationQueue:
    """Epoch-local, deduplicated cold-path requests owned by native execution."""

    __slots__ = ("_lock", "_pending", "_requested")

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: deque[RegionShapeSpecializationRequest] = deque()
        self._requested: set[str] = set()

    def enqueue(self, request: RegionShapeSpecializationRequest) -> bool:
        with self._lock:
            if request.dedup_key in self._requested:
                return False
            self._requested.add(request.dedup_key)
            self._pending.append(request)
            return True

    def peek(self) -> tuple[RegionShapeSpecializationRequest, ...]:
        with self._lock:
            return tuple(self._pending)

    def drain(self) -> tuple[RegionShapeSpecializationRequest, ...]:
        with self._lock:
            pending = tuple(self._pending)
            self._pending.clear()
            return pending

    def acknowledge(self, dedup_keys: tuple[str, ...]) -> tuple[str, ...]:
        """Remove exactly the cold-path requests confirmed by the coordinator.

        Acknowledge is deliberately all-or-nothing.  A stale parent must not
        be able to consume a prefix of a worker's pending requests after the
        worker has moved to another immutable plan epoch.
        """

        if not dedup_keys:
            raise ValueError("shape-specialization acknowledgement cannot be empty")
        if len(dedup_keys) != len(set(dedup_keys)):
            raise ValueError("shape-specialization acknowledgement contains duplicate keys")
        with self._lock:
            pending_keys = {request.dedup_key for request in self._pending}
            unknown = sorted(set(dedup_keys) - pending_keys)
            if unknown:
                raise ValueError(
                    "shape-specialization acknowledgement has no matching pending "
                    f"request(s): {unknown}"
                )
            acknowledged = set(dedup_keys)
            self._pending = deque(
                request
                for request in self._pending
                if request.dedup_key not in acknowledged
            )
        return dedup_keys


def _region_shape_specialization_request(
    compilation: "WorkloadCompilation",
    artifact: "RegionCallableArtifact",
    context: PhysicalExecutionContext,
    inputs: Mapping[str, torch.Tensor],
) -> RegionShapeSpecializationRequest:
    """Build a cold-path request solely from signed Region identities.

    Tensor metadata is observed at the CUDA resource boundary; all plan
    identities come from the immutable coordinator compilation rather than
    process-local callable or graph identities.
    """

    physical = compilation.physical
    if physical is None:
        raise RuntimeError("shape specialization requires a physical Region plan")
    if context.plan_epoch != physical.plan_epoch:
        raise RuntimeError("shape specialization context epoch differs from compilation")
    try:
        region = next(
            candidate
            for candidate in compilation.decision.execution_regions
            if candidate.region_id == artifact.region_id
        )
    except StopIteration as exc:
        raise RuntimeError("shape specialization artifact has no signed Region") from exc
    if artifact.artifact_key != region.artifact_key:
        raise RuntimeError("shape specialization artifact key differs from signed Region")
    if not compilation.decision.signature:
        raise RuntimeError("shape specialization requires a signed decision")
    if set(inputs) != set(artifact.input_keys):
        raise RuntimeError("shape specialization inputs differ from signed Region bindings")

    observed_inputs = tuple(
        sorted(
            (
                RegionShapeSpecializationInput(
                    binding_id=binding_id,
                    shape=tuple(int(dimension) for dimension in inputs[binding_id].shape),
                    dtype=str(inputs[binding_id].dtype),
                    device=str(inputs[binding_id].device),
                )
                for binding_id in artifact.input_keys
            ),
            key=lambda item: item.binding_id,
        )
    )
    return RegionShapeSpecializationRequest(
        base_cache_key=compilation.cache_key,
        base_signature=compilation.decision.signature,
        observed_plan_epoch=physical.plan_epoch,
        region_id=region.region_id,
        artifact_key=region.artifact_key,
        inputs=observed_inputs,
    )


@dataclass(frozen=True, slots=True)
class StateVersionBinding:
    """Process-local version source for one Rust-typed state resource."""

    state: ScheduleState
    snapshot: Callable[[int], Mapping[Hashable, int]]

class StateVersionRegistry:
    """Audits signed Region effects against concrete process-local versions."""

    __slots__ = ("_bindings", "_observed")

    _MUTATING = frozenset(
        {
            ScheduleEffectKind.STATE_WRITE,
            ScheduleEffectKind.STATE_TRANSITION,
            ScheduleEffectKind.VERSION_PUBLISH,
        }
    )

    def __init__(
        self,
        states: tuple[ScheduleState, ...] = (),
        bindings: Mapping[str, StateVersionBinding] = MappingProxyType({}),
        effects: tuple[ScheduleEffect, ...] = (),
    ) -> None:
        state_by_resource = {state.resource: state for state in states}
        required = {
            effect.resource
            for effect in effects
            if effect.kind is not ScheduleEffectKind.EXTERN_IO
        }
        supplied = set(bindings)
        missing = sorted(required - supplied)
        unexpected = sorted(supplied - required)
        if missing or unexpected:
            raise RuntimeError(
                "native Region state/version registry binding coverage mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        normalized: dict[str, StateVersionBinding] = {}
        for resource, binding in bindings.items():
            expected = state_by_resource.get(resource)
            if expected is None or binding.state != expected:
                raise RuntimeError(
                    f"state/version descriptor mismatch for resource {resource!r}"
                )
            normalized[resource] = binding
        self._bindings = MappingProxyType(normalized)
        self._observed: dict[tuple[str, Hashable], int] = {}

    @staticmethod
    def _snapshot(
        binding: StateVersionBinding,
        step_id: int,
    ) -> dict[Hashable, int]:
        raw = binding.snapshot(step_id)
        result: dict[Hashable, int] = {}
        for key, version in raw.items():
            if version < 0:
                raise ValueError("state versions must be non-negative ints")
            result[key] = version
        if binding.state.scope != "request" and len(result) != 1:
            raise RuntimeError(
                f"state resource {binding.state.resource!r} requires one version slot"
            )
        return result

    def before(
        self,
        effects: tuple[ScheduleEffect, ...],
        step_id: int,
    ) -> Mapping[str, Mapping[Hashable, int]]:
        snapshots: dict[str, Mapping[Hashable, int]] = {}
        for effect in effects:
            if effect.kind is ScheduleEffectKind.EXTERN_IO:
                continue
            if effect.resource not in snapshots:
                binding = self._bindings[effect.resource]
                snapshot = self._snapshot(binding, step_id)
                snapshots[effect.resource] = MappingProxyType(snapshot)
                self._observed.update(
                    ((effect.resource, key), version)
                    for key, version in snapshot.items()
                )
        return MappingProxyType(snapshots)

    def after(
        self,
        effects: tuple[ScheduleEffect, ...],
        step_id: int,
        before: Mapping[str, Mapping[Hashable, int]],
    ) -> None:
        mutation_counts = Counter(
            effect.resource for effect in effects if effect.kind in self._MUTATING
        )
        conditional_counts = Counter(
            effect.resource
            for effect in effects
            if effect.kind is ScheduleEffectKind.STATE_TRANSITION_MAYBE
        )
        resources = {
            effect.resource
            for effect in effects
            if effect.kind is not ScheduleEffectKind.EXTERN_IO
        }
        for resource in resources:
            current = self._snapshot(self._bindings[resource], step_id)
            previous = before[resource]
            if set(current) != set(previous):
                raise RuntimeError(
                    f"state resource {resource!r} changed version-slot identity within a Region"
                )
            increment = mutation_counts[resource]
            conditional = conditional_counts[resource]
            # Each conditional transition may independently fire or skip per
            # execution (the signed graph cannot prove which), so a slot may
            # advance by any count between the strict increment and strict +
            # conditional — every subset of the signed conditional
            # transitions is legal, and never more than all of them.
            invalid = {
                key: (previous[key], current[key])
                for key in current
                if not increment <= current[key] - previous[key] <= increment + conditional
            }
            if invalid:
                action = "mutation" if increment or conditional else "read"
                raise RuntimeError(
                    f"state resource {resource!r} violated signed {action} version effect: "
                    f"{invalid!r}"
                )
            self._observed.update(
                ((resource, key), version) for key, version in current.items()
            )

    def observed_version(self, resource: str, key: Hashable) -> int:
        try:
            return self._observed[(resource, key)]
        except KeyError as exc:
            raise KeyError((resource, key)) from exc

    def observed_versions(self) -> tuple[tuple[str, Hashable, int], ...]:
        """Return a stable diagnostic snapshot without exposing mutable state."""

        return tuple(
            (resource, key, version)
            for (resource, key), version in sorted(
                self._observed.items(),
                key=lambda item: (
                    item[0][0],
                    type(item[0][1]).__name__,
                    repr(item[0][1]),
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class NoExecutionLifecycle:
    """An execution with no local training lifecycle."""

    def read_and_reset_stats(self, sync: bool) -> "TrainingStats":
        from tllm.contracts.consumer_stats import TrainingStats

        _ = sync
        return TrainingStats(loss_avg=0.0, loss_count=0)

    def read_and_reset_per_request_stats(
        self, sync: bool
    ) -> Mapping[int, "TrainingStats"]:
        _ = sync
        return MappingProxyType({})

    def is_enabled(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class ExecutionLifecycleBinding:
    """The complete lifecycle capability of one native execution."""

    stats: Callable[[bool], "TrainingStats"]
    is_enabled: Callable[[], bool]
    synchronize: Callable[[], None]
    per_request_stats: PerRequestStatsService = field(
        default_factory=NoPerRequestStats
    )

    def read_and_reset_stats(self, sync: bool) -> "TrainingStats":
        return self.stats(bool(sync))

    def read_and_reset_per_request_stats(
        self, sync: bool
    ) -> Mapping[int, "TrainingStats"]:
        match self.per_request_stats:
            case PerRequestStats(read=reader):
                return reader(bool(sync))
            case NoPerRequestStats():
                return MappingProxyType({})


def _combined_training_stats(
    values: tuple["TrainingStats", ...],
) -> "TrainingStats":
    from tllm.contracts.consumer_stats import TrainingStats

    count = sum(value.loss_count for value in values)
    total = sum(value.loss_avg * value.loss_count for value in values)
    return TrainingStats(
        loss_avg=0.0 if count == 0 else total / count,
        loss_count=count,
    )


@dataclass(frozen=True, slots=True)
class CompositeExecutionLifecycleBinding:
    """Independent lifecycle services composed without shared authority."""

    bindings: tuple[ExecutionLifecycleBinding, ...]

    def read_and_reset_stats(self, sync: bool) -> "TrainingStats":
        return _combined_training_stats(
            tuple(binding.read_and_reset_stats(sync) for binding in self.bindings)
        )

    def read_and_reset_per_request_stats(
        self, sync: bool
    ) -> Mapping[int, "TrainingStats"]:
        grouped: dict[int, list["TrainingStats"]] = {}
        for binding in self.bindings:
            for key, value in binding.read_and_reset_per_request_stats(sync).items():
                grouped.setdefault(key, []).append(value)
        return MappingProxyType(
            {
                key: _combined_training_stats(tuple(values))
                for key, values in grouped.items()
            }
        )

    def is_enabled(self) -> bool:
        return any(binding.is_enabled() for binding in self.bindings)

    def synchronize(self) -> None:
        for binding in self.bindings:
            binding.synchronize()


ExecutionLifecycle = (
    NoExecutionLifecycle
    | ExecutionLifecycleBinding
    | CompositeExecutionLifecycleBinding
)


@dataclass(frozen=True, slots=True)
class WorkloadExecutionBindings:
    """Process-local callables that are deliberately absent from signed JSON."""

    declarations: tuple["WorkloadDeclaration", ...]
    ports: RuntimePortRegistry = field(default_factory=RuntimePortRegistry)
    state_versions: Mapping[str, StateVersionBinding] = field(
        default_factory=dict[str, StateVersionBinding]
    )
    sampler: SamplerBinding = field(default_factory=NoSamplerBinding)
    diagnostics_readers: Mapping[
        str, Callable[[], Mapping[str, int | float]]
    ] = field(
        default_factory=dict[str, Callable[[], Mapping[str, int | float]]]
    )
    lifecycle: ExecutionLifecycle = field(default_factory=NoExecutionLifecycle)
    residual_paths: Mapping["ResidualLocator", str] = field(
        default_factory=dict["ResidualLocator", str]
    )
    backend_region_binders: Mapping[str, "BackendRegionBinderSlot"] = field(
        default_factory=dict[str, "BackendRegionBinderSlot"]
    )
    parameter_overlays: Mapping[OverlayChannel, OverlayVersionStore] = field(
        default_factory=dict[OverlayChannel, OverlayVersionStore]
    )
    # These are immutable live resource snapshots, not signed payloads.  The
    # local execution gate serializes them immediately before artifacts bind.
    static_binding_entries: tuple["StaticBackendBindingEntry", ...] = ()

    def __post_init__(self) -> None:
        if not self.declarations:
            raise ValueError("declarations must not be empty")
        if any(not resource for resource in self.state_versions):
            raise ValueError("state-version resource names must not be empty")
        object.__setattr__(
            self, "state_versions", MappingProxyType(dict(self.state_versions))
        )
        if any(not workload_id for workload_id in self.diagnostics_readers):
            raise ValueError("diagnostic workload IDs must not be empty")
        object.__setattr__(
            self,
            "diagnostics_readers",
            MappingProxyType(dict(self.diagnostics_readers)),
        )
        if any(not path for path in self.residual_paths.values()):
            raise ValueError("residual paths must not be empty")
        object.__setattr__(
            self, "residual_paths", MappingProxyType(dict(self.residual_paths))
        )
        if any(
            not implementation_id or slot.implementation_id != implementation_id
            for implementation_id, slot in self.backend_region_binders.items()
        ):
            raise ValueError(
                "backend binder IDs must match their binding slots"
            )
        object.__setattr__(
            self,
            "backend_region_binders",
            MappingProxyType(dict(self.backend_region_binders)),
        )
        object.__setattr__(
            self,
            "parameter_overlays",
            MappingProxyType(dict(self.parameter_overlays)),
        )


class _SerialStream:
    """Default-stream token for a serial plan with no cross-stream fences."""

    def wait_event(self, event: object) -> None:
        if torch.cuda.is_available() and isinstance(event, torch.cuda.Event):
            torch.cuda.current_stream().wait_event(event)


class _ImmediateEvent:
    def record(self, stream: object) -> None:
        _ = stream

    def query(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class _NoRegionValueStore:
    """A signed execution with no Region values to retain."""

    def prepare_step(self, step_id: int) -> None:
        _ = step_id

    def finish_step(self, step_id: int) -> None:
        _ = step_id

    @property
    def retained_value_count(self) -> int:
        return 0


@dataclass(slots=True)
class NativeWorkloadExecution:
    """One complete, immutable workload plan bound to process-local handles."""

    compilation: "WorkloadCompilation"
    executor: PhysicalExecutor
    completion_anchor: str
    _owned_consumers: tuple[object, ...]
    _workload_instances: tuple[WorkloadPortOwner, ...] = ()
    _anchor_nodes: Mapping[str, tuple[NodeKey, ...]] = field(
        default_factory=dict[str, tuple[NodeKey, ...]]
    )
    _event_nodes: Mapping[BackendObservableEvent, tuple[NodeKey, ...]] = field(
        default_factory=dict[BackendObservableEvent, tuple[NodeKey, ...]]
    )
    _value_store: "_RegionValueStore | _NoRegionValueStore" = field(
        default_factory=_NoRegionValueStore
    )
    _state_versions: StateVersionRegistry = field(
        default_factory=StateVersionRegistry
    )
    _artifact_report: tuple[tuple[str, str], ...] = ()
    _sampler: SamplerBinding = field(default_factory=NoSamplerBinding)
    _diagnostics_readers: Mapping[
        str, Callable[[], Mapping[str, int | float]]
    ] = field(
        default_factory=dict[
            str, Callable[[], Mapping[str, int | float]]
        ]
    )
    _lifecycle: ExecutionLifecycle = field(default_factory=NoExecutionLifecycle)
    _backend_region_binders: Mapping[str, "BackendRegionBinderSlot"] = field(
        default_factory=dict[str, "BackendRegionBinderSlot"]
    )
    parameter_overlays: Mapping[OverlayChannel, OverlayVersionStore] = field(
        default_factory=dict[OverlayChannel, OverlayVersionStore]
    )
    _requires_decode_hidden: bool = False
    _shape_specializations: _NativeShapeSpecializationQueue = field(
        default_factory=_NativeShapeSpecializationQueue
    )
    _prepared_step: int = -1
    _executed_nodes: set[NodeKey] = field(default_factory=set[NodeKey])
    _side_streams: tuple[object, ...] = ()
    _cost_probe: RegionCostProbe = field(default_factory=RegionCostProbe)

    def _join_side_streams(self) -> None:
        """Order the main stream after all side-stream work enqueued so far.

        Called at every signed advance and at step retirement: side Regions
        overlap with the main workload between two signed events, and any
        main-path consumption at the next event (sinks, backend binders)
        observes their results.  This is an async event wait, not a host
        synchronization.
        """
        if not self._side_streams or not torch.cuda.is_available():
            return
        current = torch.cuda.current_stream()
        for stream in self._side_streams:
            current.wait_stream(cast(torch.cuda.Stream, stream))

    def owns_consumer(self, consumer: object) -> bool:
        return any(owner is consumer for owner in self._owned_consumers)

    def workload_instance(
        self, workload_type: type[WorkloadInstanceT]
    ) -> WorkloadInstanceT:
        matches = tuple(
            instance
            for instance in self._workload_instances
            if type(instance) is workload_type
        )
        if len(matches) != 1:
            raise LookupError(
                f"active execution has {len(matches)} instances of "
                f"{workload_type.__qualname__}"
            )
        return cast(WorkloadInstanceT, matches[0])

    def prepare_port(self, key: str, transaction: str, value: object) -> None:
        for workload in self._workload_instances:
            if workload.owns_port(key):
                workload.prepare_port(key, transaction, value)
                return
        raise KeyError(key)

    def commit_port(self, key: str, transaction: str) -> int:
        for workload in self._workload_instances:
            if workload.owns_port(key):
                return workload.commit_port(key, transaction)
        raise KeyError(key)

    def abort_port(self, key: str, transaction: str) -> None:
        for workload in self._workload_instances:
            if workload.owns_port(key):
                workload.abort_port(key, transaction)
                return
        raise KeyError(key)

    def port_transaction_status(
        self, key: str, transaction: str
    ) -> PortTransactionStatus:
        for workload in self._workload_instances:
            if workload.owns_port(key):
                return workload.port_transaction_status(key, transaction)
        raise KeyError(key)

    def acknowledge_remote_port_consumed(
        self, key: str, transaction: str
    ) -> None:
        for workload in self._workload_instances:
            if workload.owns_port(key):
                workload.acknowledge_remote_port_consumed(key, transaction)
                return
        raise KeyError(key)

    def abort_port_transaction(
        self, keys: tuple[str, ...], transaction: str
    ) -> bool:
        if not keys:
            raise ValueError("staged port transaction must contain at least one port")
        statuses = tuple(
            self.port_transaction_status(key, transaction) for key in keys
        )
        if all(status is PortTransactionStatus.PENDING for status in statuses):
            for key in keys:
                self.abort_port(key, transaction)
            return True
        if all(status is PortTransactionStatus.CONSUMED for status in statuses):
            return False
        raise RuntimeError("staged port transaction members have diverged")

    def confirm_port_transaction_consumed(
        self, keys: tuple[str, ...], transaction: str
    ) -> None:
        if not keys:
            raise ValueError("staged port transaction must contain at least one port")
        statuses = tuple(
            self.port_transaction_status(key, transaction) for key in keys
        )
        if not all(
            status is PortTransactionStatus.CONSUMED for status in statuses
        ):
            raise RuntimeError("staged port transaction was not fully consumed")

    def drain_output(self, key: str) -> tuple[object, ...]:
        for workload in self._workload_instances:
            if workload.owns_output(key):
                return workload.drain_output(key)
        raise KeyError(key)

    def stage_request_finished(self, batch: RequestFinishedBatch) -> int:
        return sum(
            workload.stage_request_finished(batch)
            for workload in self._workload_instances
        )

    def stage_dense_operand_captured(
        self, batch: DenseOperandCaptureBatch
    ) -> int:
        return sum(
            workload.stage_dense_operand_captured(batch)
            for workload in self._workload_instances
        )

    @property
    def backend_region_binders(self) -> Mapping[str, "BackendRegionBinderSlot"]:
        """Read-only local backend binder registry for signed extern Regions."""

        return self._backend_region_binders

    def region_cost_report(self, *, block: bool = False) -> RegionCostReport:
        """Read-only observed region cost snapshot (Phase D backfill).

        Diagnostics only: the signed Decision (including its
        ``Cost.main_path_extension_ns`` placeholder) is never mutated, and no
        measurement feeds back into the compiler.
        """

        signed = self.compilation.decision.cost.main_path_extension_ns
        return self._cost_probe.report(
            signed_main_path_extension_ns=signed, block=block
        )

    def backend_region_binder_stages(
        self, backend_event: BackendObservableEvent
    ) -> tuple["BackendRegionBinderStage", ...]:
        """Return Rust-signed backend binders for one physical backend event."""

        stages: list["BackendRegionBinderStage"] = []
        for advance in self.compilation.decision.runtime_advance_plan:
            if advance.backend_event is not backend_event:
                continue
            stages.extend(advance.backend_region_binder_stages)
        return tuple(stages)

    def prepare_step(self, step_id: int) -> None:
        step = int(step_id)
        if step < 0:
            raise ValueError("native workload step id must be non-negative")
        if self._prepared_step == step:
            return
        if self._prepared_step >= 0:
            self.finish_step(self._prepared_step)
        self.executor.prepare_step(step)
        self._value_store.prepare_step(step)
        self._prepared_step = step
        self._executed_nodes.clear()

    def finish_step(self, step_id: int) -> None:
        """Retire one prepared step without releasing epoch-owned artifacts."""

        step = int(step_id)
        if self._prepared_step < 0:
            return
        if self._prepared_step != step:
            raise RuntimeError("cannot retire a native workload step from another epoch")
        self._join_side_streams()
        self.executor.finish_step(step)
        self._value_store.finish_step(step)
        self._cost_probe.harvest()
        self._prepared_step = -1
        self._executed_nodes.clear()

    def advance_anchor(self, anchor: str) -> bool:
        """Compatibility entry point for an already-signed timeline anchor."""

        anchor = str(anchor)
        nodes = self._anchor_nodes.get(anchor)
        if nodes is None:
            return False
        return self._advance_nodes(nodes)

    def advance_backend_event(self, backend_event: BackendObservableEvent) -> bool:
        """Advance only the Region nodes signed for one backend event."""

        nodes = self._event_nodes.get(backend_event)
        if nodes is None:
            raise RuntimeError(
                "backend event "
                f"{backend_event.value!r} is not signed for this native workload"
            )
        # A signed timeline may expose an observable event that advances no
        # Region. It is an intentional lifecycle no-op, not an unsigned hook.
        # The side-stream join is still owed here: main-path consumers at
        # this point in the timeline (e.g. the sampler provider reached via
        # dispatch_sampler_pre_draw) must observe completed side work even
        # when the event itself advances zero Region nodes.
        if not nodes:
            self._join_side_streams()
            return False
        return self._advance_nodes(nodes)

    def _advance_nodes(self, nodes: tuple[NodeKey, ...]) -> bool:
        if self._prepared_step < 0:
            # vLLM kernel warmup calls sampler methods without opening a
            # logical engine step. It has no workload values to execute.
            return False
        # Join side-stream work from earlier signed events before this
        # event's nodes or main-path consumers observe their outputs.
        self._join_side_streams()
        pending = tuple(node for node in nodes if node not in self._executed_nodes)
        if not pending:
            return False
        self.executor.execute_nodes(self._prepared_step, pending)
        self._executed_nodes.update(pending)
        return True

    def close(self) -> None:
        if self._prepared_step >= 0:
            self.finish_step(self._prepared_step)

    @property
    def retained_value_count(self) -> int:
        return self._value_store.retained_value_count

    @property
    def state_versions(self) -> StateVersionRegistry:
        return self._state_versions

    def state_version_report(self) -> tuple[tuple[str, object, int], ...]:
        """Project observed versions onto the closed JSON control-plane key set."""

        report: list[tuple[str, object, int]] = []
        for resource, key, version in self._state_versions.observed_versions():
            if key is not None and (
                isinstance(key, bool) or not isinstance(key, (str, int))
            ):
                raise TypeError(
                    "state version report keys must be None, str, or non-bool int"
                )
            report.append((resource, key, version))
        return tuple(report)

    @property
    def artifact_report(self) -> tuple[tuple[str, str], ...]:
        return self._artifact_report

    @property
    def plan_epoch(self) -> int:
        physical = self.compilation.physical
        if physical is None:
            raise RuntimeError("native workload execution requires a physical plan epoch")
        return physical.plan_epoch

    @property
    def requires_decode_hidden(self) -> bool:
        """Whether the signed Region plan consumes decode residual inputs."""

        return self._requires_decode_hidden

    def sampler_modifier_provider(self) -> "SamplerModifierProvider | None":
        match self._sampler:
            case SamplerRuntimeBinding(provider=provider):
                return provider
            case NoSamplerBinding():
                return None

    def read_resource_diagnostics(self) -> Mapping[str, Mapping[str, int | float]]:
        """Return a validated, JSON-safe snapshot of local workload resources.

        Resource diagnostics deliberately remain outside the signed compilation:
        they describe the active worker instance only and cannot influence
        scheduling, artifact identity, or cache selection.
        """

        result: dict[str, Mapping[str, int | float]] = {}
        for workload_id, reader in self._diagnostics_readers.items():
            raw = reader()
            normalized: dict[str, int | float] = {}
            for name, value in raw.items():
                if not name:
                    raise TypeError(
                        "native diagnostic names must be non-empty strings"
                    )
                if type(value) not in (int, float):
                    raise TypeError(
                        "native diagnostic values must be int or float"
                    )
                if type(value) is float and not isfinite(value):
                    raise ValueError("native diagnostic float values must be finite")
                normalized[name] = value
            result[workload_id] = MappingProxyType(dict(sorted(normalized.items())))
        return MappingProxyType(dict(sorted(result.items())))

    def read_and_reset_stats(self, *, sync: bool = True) -> "TrainingStats":
        return self._lifecycle.read_and_reset_stats(bool(sync))

    def read_and_reset_per_request_stats(
        self, *, sync: bool = True
    ) -> Mapping[int, "TrainingStats"]:
        return self._lifecycle.read_and_reset_per_request_stats(bool(sync))

    def set_enabled(self, enabled: bool) -> None:
        _ = enabled
        raise RuntimeError(
            "native workload execution is immutable; request a coordinator-owned "
            "cold epoch transition instead"
        )

    def synchronize(self) -> None:
        self._lifecycle.synchronize()

    def read_workgraph_runtime_stats(self) -> "WorkGraphRuntimeStats":
        return runtime_stats_from_executor(self.executor.stats)

    def peek_shape_specialization_requests(
        self,
    ) -> tuple[RegionShapeSpecializationRequest, ...]:
        """Inspect pending Region specializations without crossing a safe point."""

        return self._shape_specializations.peek()

    def drain_shape_specialization_requests(
        self,
    ) -> tuple[RegionShapeSpecializationRequest, ...]:
        """Consume pending cold-path requests at an explicit execution safe point."""

        return self._shape_specializations.drain()

    def acknowledge_shape_specialization_requests(
        self, dedup_keys: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Confirm parent-side epoch publication without draining unrelated requests."""

        return self._shape_specializations.acknowledge(dedup_keys)


@dataclass(frozen=True, slots=True)
class NoStagedExecution:
    pass


@dataclass(frozen=True, slots=True)
class StagedExecution:
    execution: NativeWorkloadExecution
    activated: bool


StagedExecutionState = NoStagedExecution | StagedExecution


class NativeWorkloadExecutionStore:
    """Safe-point publication and retirement for native Region executors."""

    __slots__ = (
        "_active_steps",
        "_current",
        "_retired",
        "_staged",
    )

    def __init__(self, initial: NativeWorkloadExecution) -> None:
        self._current = initial
        self._staged: StagedExecutionState = NoStagedExecution()
        self._active_steps: dict[int, NativeWorkloadExecution] = {}
        self._retired: list[NativeWorkloadExecution] = []

    @property
    def current(self) -> NativeWorkloadExecution:
        return self._current

    @property
    def staged(self) -> StagedExecutionState:
        return self._staged

    @property
    def retired_count(self) -> int:
        return len(self._retired)

    @property
    def staged_activated(self) -> bool:
        match self._staged:
            case StagedExecution(activated=activated):
                return activated
            case NoStagedExecution():
                return False

    @property
    def next_execution(self) -> NativeWorkloadExecution:
        match self._staged:
            case StagedExecution(execution=execution, activated=True):
                return execution
            case StagedExecution() | NoStagedExecution():
                return self._current

    def stage(
        self,
        execution: NativeWorkloadExecution,
        *,
        activation_required: bool = False,
    ) -> None:
        match self._staged:
            case StagedExecution():
                raise RuntimeError("a native workload epoch is already staged")
            case NoStagedExecution():
                pass
        if execution.plan_epoch <= self._current.plan_epoch:
            raise ValueError("staged native workload epoch must be strictly newer")
        self._staged = StagedExecution(
            execution, activated=not bool(activation_required)
        )

    def activate_staged(self, plan_epoch: int) -> bool:
        match self._staged:
            case NoStagedExecution():
                return False
            case StagedExecution(execution=execution):
                if execution.plan_epoch != int(plan_epoch):
                    raise RuntimeError("staged native workload epoch identity mismatch")
                self._staged = StagedExecution(execution, activated=True)
                return True

    def abort_staged(self) -> bool:
        match self._staged:
            case NoStagedExecution():
                return False
            case StagedExecution(execution=execution):
                self._staged = NoStagedExecution()
                execution.close()
                return True

    def prepare_step(self, step_id: int) -> NativeWorkloadExecution:
        step = int(step_id)
        if step < 0:
            raise ValueError("native workload step id must be non-negative")
        active = self._active_steps.get(step)
        if active is not None:
            return active
        if self._active_steps:
            raise RuntimeError("prior native workload steps must retire before prepare")
        match self._staged:
            case StagedExecution(execution=execution, activated=True):
                previous = self._current
                self._current = execution
                self._staged = NoStagedExecution()
                self._retired.append(previous)
            case StagedExecution() | NoStagedExecution():
                pass
        self._current.prepare_step(step)
        self._active_steps[step] = self._current
        return self._current

    def finish_step(self, step_id: int) -> bool:
        step = int(step_id)
        execution = self._active_steps.pop(step, None)
        if execution is None:
            return False
        execution.finish_step(step)
        return True

    def collect_retired(self) -> int:
        active_ids = {id(execution) for execution in self._active_steps.values()}
        ready = tuple(
            execution
            for execution in self._retired
            if id(execution) not in active_ids
        )
        if not ready:
            return 0
        ready_ids = {id(execution) for execution in ready}
        self._retired = [
            execution
            for execution in self._retired
            if id(execution) not in ready_ids
        ]
        for execution in ready:
            execution.close()
        return len(ready)

    def close(self) -> None:
        match self._staged:
            case StagedExecution(execution=staged):
                executions = (self._current, staged, *self._retired)
            case NoStagedExecution():
                executions = (self._current, *self._retired)
        self._active_steps.clear()
        self._staged = NoStagedExecution()
        self._retired.clear()
        seen: set[int] = set()
        for execution in executions:
            identity = id(execution)
            if identity in seen:
                continue
            seen.add(identity)
            execution.close()

class _RegionValueStore:
    """Step-scoped binding and lifetime owner for Region boundary values."""

    __slots__ = (
        "_runtime",
        "_providers",
        "_sinks",
        "_hidden_inputs",
        "_materializations",
        "_state_versions",
        "_lifecycle",
        "_values",
        "_step_id",
    )

    def __init__(
        self,
        *,
        runtime: "ResidualRuntimeState",
        providers: Mapping[str, RuntimeInputBinding[object]],
        sinks: Mapping[str, RuntimeOutputBinding[object]],
        artifacts: tuple["RegionCallableArtifact", ...],
        materializations: _SignedMaterializationBindings,
        state_versions: StateVersionRegistry,
        lifecycle: ExecutionLifecycle,
    ) -> None:
        self._runtime = runtime
        self._state_versions = state_versions
        self._lifecycle = lifecycle
        self._providers = providers
        self._sinks = sinks
        self._materializations = materializations
        self._hidden_inputs = MappingProxyType(
            {
                binding.binding_id: binding
                for artifact in artifacts
                for binding in artifact.input_bindings
                if binding.kind is RegionInputKind.HIDDEN
            }
        )
        self._values: dict[str, object] = {}
        self._step_id = -1

    @property
    def retained_value_count(self) -> int:
        return sum(value is not SKIPPED for value in self._values.values())

    def prepare_step(self, step_id: int) -> None:
        if self._step_id >= 0 and self._step_id != step_id:
            raise RuntimeError("Region values from the prior step were not retired")
        self._step_id = step_id
        self._values.clear()

    def finish_step(self, step_id: int) -> None:
        if self._step_id not in {-1, step_id}:
            raise RuntimeError("cannot retire Region values for another step")
        self._values.clear()
        self._step_id = -1

    def execute(
        self,
        artifact: "RegionCallableArtifact",
        context: PhysicalExecutionContext,
    ) -> None:
        if not self._lifecycle.is_enabled():
            self.commit_outputs(
                artifact,
                context,
                {key: SKIPPED for key in artifact.output_keys},
            )
            return
        inputs = self.resolve_inputs(artifact, context)
        state_before = self._state_versions.before(
            artifact.effects, context.step_id
        )
        if any(value is SKIPPED for value in inputs.values()):
            outputs = {key: SKIPPED for key in artifact.output_keys}
        else:
            outputs = artifact.execute(inputs)
            self._state_versions.after(
                artifact.effects,
                context.step_id,
                state_before,
            )
        self.commit_outputs(artifact, context, outputs)

    def is_enabled(self) -> bool:
        return self._lifecycle.is_enabled()

    def resolve_inputs(
        self,
        artifact: "RegionCallableArtifact",
        context: PhysicalExecutionContext,
    ) -> Mapping[str, object]:
        if self._step_id != context.step_id:
            raise RuntimeError("Region value store is not prepared for this step")
        inputs: dict[str, object] = {}
        for binding in artifact.input_bindings:
            if binding.kind is RegionInputKind.MATERIALIZED:
                self._materializations.input_binding_for(
                    artifact.region_id, binding.binding_id
                )
                if binding.binding_id not in self._values:
                    raise RuntimeError(
                        f"ExecutionRegion {artifact.region_id!r} has no materialized input {binding.binding_id!r}"
                    )
                inputs[binding.binding_id] = self._values[binding.binding_id]
            elif binding.kind is RegionInputKind.STEP_ID:
                inputs[binding.binding_id] = context.step_id
            elif binding.kind in (
                RegionInputKind.PROVIDED,
                RegionInputKind.STAGED,
                RegionInputKind.SAMPLER_REQUEST_IDS,
                RegionInputKind.SAMPLER_SAMPLE_IDXS,
                RegionInputKind.SAMPLER_LOGITS,
                RegionInputKind.SAMPLED_TOKENS,
            ):
                inputs[binding.binding_id] = self._providers[
                    binding.source
                ].read(context.step_id)
            elif binding.kind is RegionInputKind.HIDDEN:
                inputs[binding.binding_id] = _hidden_value(
                    self._runtime,
                    self._hidden_inputs[binding.binding_id],
                )
            else:
                raise AssertionError(f"unhandled signed Region input kind {binding.kind!r}")
        return inputs

    def commit_outputs(
        self,
        artifact: "RegionCallableArtifact",
        context: PhysicalExecutionContext,
        outputs: Mapping[str, object],
    ) -> None:
        if self._step_id != context.step_id:
            raise RuntimeError("Region value store is not prepared for this step")
        if set(outputs) != set(artifact.output_keys):
            raise RuntimeError(
                f"ExecutionRegion {artifact.region_id!r} produced invalid output bindings"
            )
        for binding in artifact.output_bindings:
            value = outputs[binding.binding_id]
            if binding.kind in (
                RegionOutputKind.SINK,
                RegionOutputKind.HOST,
                RegionOutputKind.SAMPLER,
            ):
                if value is not SKIPPED:
                    self._sinks[binding.target].publish(value, context.step_id)
            elif binding.kind is RegionOutputKind.MATERIALIZED:
                self._materializations.output_binding_for(
                    artifact.region_id, binding.binding_id
                )
                self._values[binding.binding_id] = value
            else:
                raise AssertionError(f"unhandled signed Region output kind {binding.kind!r}")

        for binding in artifact.input_bindings:
            if binding.kind is not RegionInputKind.MATERIALIZED:
                continue
            self._materializations.input_binding_for(
                artifact.region_id, binding.binding_id
            )
            key = binding.binding_id
            # Each materialization record names exactly one Rust-signed
            # consumer Region. Release that explicit boundary after its
            # consumer executes; Python does not derive a liveness count.
            self._values.pop(key, None)


class _RegionCudaGraph:
    """One lazily captured, address-stable CUDA graph for a signed Region."""

    __slots__ = (
        "_artifact",
        "_store",
        "_graph",
        "_stream",
        "_staging",
        "_input_specs",
        "_outputs",
        "_shape_specialization_request",
        "_preflight_identity",
        "_preflight_inputs",
        "_preflight_disabled",
        "_preflight_miss",
        "_cost_probe",
        "_stream_label",
    )

    def __init__(
        self,
        artifact: "RegionCallableArtifact",
        store: _RegionValueStore,
        shape_specialization_request: Callable[
            [PhysicalExecutionContext, Mapping[str, torch.Tensor]],
            RegionShapeSpecializationRequest,
        ]
        | None = None,
        *,
        cost_probe: RegionCostProbe | None = None,
        stream_label: str = "default",
    ) -> None:
        if artifact.effects:
            raise RuntimeError(
                f"capture-safe Region {artifact.region_id!r} cannot contain state effects"
            )
        self._artifact = artifact
        self._store = store
        self._graph: torch.cuda.CUDAGraph | None = None
        self._stream: torch.cuda.Stream | None = None
        self._staging: dict[str, torch.Tensor] = {}
        self._input_specs: dict[
            str, tuple[tuple[int, ...], torch.dtype, torch.device]
        ] = {}
        self._outputs: tuple[torch.Tensor, ...] = ()
        self._shape_specialization_request = shape_specialization_request
        self._preflight_identity: tuple[int, int] | None = None
        self._preflight_inputs: Mapping[str, torch.Tensor] | None = None
        self._preflight_disabled = False
        self._preflight_miss: RegionShapeSpecializationRequest | None = None
        self._cost_probe = cost_probe
        self._stream_label = str(stream_label)

    def replay(self) -> None:
        raise RuntimeError("Region CUDA Graph replay requires execution context")

    @staticmethod
    def _tensor_inputs(
        artifact: "RegionCallableArtifact",
        inputs: Mapping[str, object],
    ) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for key in artifact.input_keys:
            value = inputs[key]
            if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
                raise RuntimeError(
                    f"capture-safe Region {artifact.region_id!r} requires CUDA Tensor inputs"
                )
            tensors[key] = value
        return tensors

    @staticmethod
    def _tensor_outputs(
        artifact: "RegionCallableArtifact",
        outputs: object,
    ) -> tuple[torch.Tensor, ...]:
        if not isinstance(outputs, tuple):
            raise RuntimeError("captured Region returned an invalid output tuple")
        raw_outputs = cast(tuple[object, ...], outputs)
        if len(raw_outputs) != len(artifact.output_keys):
            raise RuntimeError("captured Region returned an invalid output tuple")
        if any(not isinstance(value, torch.Tensor) for value in raw_outputs):
            raise RuntimeError(
                f"capture-safe Region {artifact.region_id!r} requires Tensor outputs"
            )
        return tuple(cast(torch.Tensor, value) for value in raw_outputs)

    def _capture(self, inputs: Mapping[str, torch.Tensor]) -> None:
        artifact = self._artifact
        if not inputs:
            raise RuntimeError("captured Region requires at least one Tensor input")
        devices = {value.device for value in inputs.values()}
        if len(devices) != 1:
            raise RuntimeError("captured Region inputs must use one CUDA device")
        device = next(iter(devices))
        current = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        staging = {
            key: torch.empty_like(value) for key, value in inputs.items()
        }
        input_specs = {
            key: (tuple(value.shape), value.dtype, value.device)
            for key, value in inputs.items()
        }
        capture_stream.wait_stream(current)
        with torch.cuda.stream(capture_stream):
            for key, value in inputs.items():
                staging[key].copy_(value)
            warmup = artifact.callable(
                *(staging[key] for key in artifact.input_keys)
            )
            warmup_outputs = self._tensor_outputs(artifact, warmup)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            captured = artifact.callable(
                *(staging[key] for key in artifact.input_keys)
            )
        outputs = self._tensor_outputs(artifact, captured)
        with torch.cuda.stream(capture_stream):
            graph.replay()
        capture_stream.synchronize()
        # First-replay parity gate (native PG146 analogue): the captured graph
        # must reproduce the pre-capture oracle run on identical staging
        # inputs.  Any divergence fails closed before the graph is installed.
        # Approximation Regions compare under their declaration-supplied
        # criteria; exact Regions under dtype-aware defaults.
        compare_region_outputs(
            artifact.region_id,
            warmup_outputs,
            outputs,
            tolerance=artifact.parity_tolerance,
        )
        current.wait_stream(capture_stream)
        self._graph = graph
        self._stream = capture_stream
        self._staging = staging
        self._input_specs = input_specs
        self._outputs = outputs

    @staticmethod
    def _context_identity(context: PhysicalExecutionContext) -> tuple[int, int]:
        return (context.plan_epoch, context.step_id)

    def _clear_preflight(self) -> None:
        self._preflight_identity = None
        self._preflight_inputs = None
        self._preflight_disabled = False
        self._preflight_miss = None

    def preflight_context(self, context: object) -> None:
        """Resolve and validate replay inputs without CUDA submission or commit."""

        if not isinstance(context, PhysicalExecutionContext):
            raise TypeError("Region CUDA Graph requires PhysicalExecutionContext")
        identity = self._context_identity(context)
        if self._preflight_identity == identity:
            request = self._preflight_miss
            if request is not None:
                raise ShapeSpecializationMiss(request)
            return
        self._clear_preflight()
        self._preflight_identity = identity
        if not self._store.is_enabled():
            self._preflight_disabled = True
            return
        inputs = self._store.resolve_inputs(self._artifact, context)
        if any(value is SKIPPED for value in inputs.values()):
            self._preflight_disabled = True
            return
        tensors = self._tensor_inputs(self._artifact, inputs)
        if self._graph is not None:
            for key, value in tensors.items():
                spec = (tuple(value.shape), value.dtype, value.device)
                if self._input_specs.get(key) != spec:
                    request_shape_specialization = self._shape_specialization_request
                    if request_shape_specialization is None:
                        raise RuntimeError(
                            "ExecutionRegion "
                            f"{self._artifact.region_id!r} has a shape specialization "
                            "miss without a signed request binding"
                        )
                    request = request_shape_specialization(context, tensors)
                    self._preflight_miss = request
                    raise ShapeSpecializationMiss(request)
        self._preflight_inputs = tensors

    def replay_context(self, context: object) -> None:
        if not isinstance(context, PhysicalExecutionContext):
            raise TypeError("Region CUDA Graph requires PhysicalExecutionContext")
        self.preflight_context(context)
        if self._preflight_disabled:
            self._store.commit_outputs(
                self._artifact,
                context,
                {key: SKIPPED for key in self._artifact.output_keys},
            )
            self._clear_preflight()
            return
        tensors = self._preflight_inputs
        if tensors is None:
            raise RuntimeError("Region CUDA Graph replay was not preflighted")
        probe = self._cost_probe
        if self._graph is None:
            # Capture is one-time and host-visible; measure its wall time.
            timer = (
                probe.measure_host(self._artifact.region_id, "capture", self._stream_label)
                if probe is not None
                else contextlib.nullcontext()
            )
            with timer:
                self._capture(tensors)
        else:
            stream = self._stream
            graph = self._graph
            assert stream is not None and graph is not None
            current = torch.cuda.current_stream(next(iter(tensors.values())).device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                # Events recorded inside this block live on the capture stream,
                # so the reading is the graph's own GPU span, not the fencing.
                timer = (
                    probe.measure(self._artifact.region_id, "replay", self._stream_label)
                    if probe is not None
                    else contextlib.nullcontext()
                )
                with timer:
                    for key, value in tensors.items():
                        self._staging[key].copy_(value)
                    graph.replay()
            current.wait_stream(stream)
        outputs = dict(
            zip(self._artifact.output_keys, self._outputs, strict=True)
        )
        self._store.commit_outputs(self._artifact, context, outputs)
        self._clear_preflight()


def _hidden_value(
    runtime: "ResidualRuntimeState",
    binding: RegionInputBinding,
) -> object:
    from tllm.ports.residual_stream import ResidualLocator
    from tllm.runtime.ports.residual_bindings import resolved_path_for_locator

    if (
        binding.site is not HiddenSite.BLOCK_OUTPUT
        or binding.phase is not HiddenPhase.DECODE
    ):
        raise AssertionError(
            "Rust admitted a hidden locator the native vLLM executor cannot bind"
        )
    active = int(runtime.decode_count)
    if active <= 0:
        return SKIPPED
    path: str | None
    if binding.layer_kind is HiddenLayerKind.INDEX:
        assert binding.layer_index is not None
        locator = ResidualLocator(
            layer=binding.layer_index,
            site="block_output",
            phase="decode",
        )
        path = resolved_path_for_locator(runtime.residual_bindings, locator)
    elif binding.layer_kind is HiddenLayerKind.PATH:
        path = runtime.native_hidden_path_aliases.get(binding.layer)
        if path is None:
            raise RuntimeError(
                "native Region PATH hidden input requires an explicit backend "
                f"alias for {binding.layer!r}"
            )
    elif binding.layer_kind is HiddenLayerKind.WILDCARD:
        raise RuntimeError(
            "native Region wildcard hidden input requires an explicit backend binding"
        )
    else:
        raise AssertionError(
            f"unhandled signed hidden layer kind {binding.layer_kind!r}"
        )
    if not path:
        raise RuntimeError(
            f"native Region has no resolved hidden input for {binding.source!r}"
        )
    hidden = runtime.tap_decode_hidden.get(path)
    if not isinstance(hidden, torch.Tensor):
        raise RuntimeError(
            f"native Region hidden input {binding.source!r} was not materialized"
        )
    if active > int(hidden.shape[0]):
        raise RuntimeError("native Region decode row count exceeds hidden storage")
    selected = hidden[:active]
    if torch.is_inference(selected):
        # vLLM executes the model under inference_mode. A compiled Region that
        # owns autograd must receive an ordinary tensor at this explicit
        # backend boundary; cloning inside the compiled callable is too late
        # because AOTAutograd saves its external inputs first.
        with torch.inference_mode(False):
            return selected.clone()
    return selected


def _validate_hidden_resource_bindings(
    compilation: "WorkloadCompilation",
    bindings: WorkloadExecutionBindings,
) -> None:
    """Require every signed hidden input to have an install-time resource."""

    from tllm.ports.residual_stream import ResidualLocator

    residual_paths = bindings.residual_paths
    indexed: set[ResidualLocator] = set()
    paths: set[tuple[str, str, str]] = set()
    for region in compilation.decision.execution_regions:
        for binding in region.input_bindings:
            if binding.kind is not RegionInputKind.HIDDEN:
                continue
            if (
                binding.site is not HiddenSite.BLOCK_OUTPUT
                or binding.phase is not HiddenPhase.DECODE
            ):
                raise RuntimeError(
                    "native Region hidden input locator is not supported by the "
                    f"backend: {binding.source!r}"
                )
            site = binding.site.value
            phase = binding.phase.value
            if binding.layer_kind is HiddenLayerKind.INDEX:
                assert binding.layer_index is not None
                indexed.add(
                    ResidualLocator(
                        layer=binding.layer_index,
                        site=site,
                        phase=phase,
                    )
                )
            elif binding.layer_kind is HiddenLayerKind.PATH:
                paths.add((binding.layer, site, phase))
            elif binding.layer_kind is HiddenLayerKind.WILDCARD:
                raise RuntimeError(
                    "native Region wildcard hidden input has no explicit backend "
                    f"resource binding: {binding.source!r}"
                )
            else:
                raise RuntimeError(
                    "native Region hidden input has an unsupported layer kind: "
                    f"{binding.layer_kind.value!r}"
                )

    supplied = set(residual_paths)
    missing_indexed = indexed - supplied
    if missing_indexed:
        raise RuntimeError(
            "native Region missing INDEX hidden resource binding(s): "
            + ", ".join(map(str, sorted(missing_indexed, key=str)))
        )
    supplied_paths = {
        (raw_path, locator.site, locator.phase)
        for locator, raw_path in residual_paths.items()
    }
    missing_paths = paths - supplied_paths
    if missing_paths:
        raise RuntimeError(
            "native Region missing PATH hidden resource binding(s): "
            + ", ".join(
                f"{path!r}@{site}:{phase}"
                for path, site, phase in sorted(missing_paths)
            )
        )
    unexpected = tuple(
        locator
        for locator, raw_path in residual_paths.items()
        if locator not in indexed
        and (raw_path, locator.site, locator.phase) not in paths
    )
    if unexpected:
        raise RuntimeError(
            "native Region unexpected residual resource binding(s): "
            + ", ".join(map(str, sorted(unexpected, key=str)))
        )


def _local_binding_requirements(
    compilation: "WorkloadCompilation",
    bindings: WorkloadExecutionBindings,
) -> StateVersionRegistry:
    _validate_hidden_resource_bindings(compilation, bindings)
    required_backend_region_binders = {
        stage.implementation_id
        for advance in compilation.decision.runtime_advance_plan
        for stage in advance.backend_region_binder_stages
    }
    supplied_backend_region_binders = set(bindings.backend_region_binders)
    missing_backend_region_binders = sorted(
        required_backend_region_binders - supplied_backend_region_binders
    )
    unexpected_backend_region_binders = sorted(
        supplied_backend_region_binders - required_backend_region_binders
    )
    if missing_backend_region_binders or unexpected_backend_region_binders:
        raise RuntimeError(
            "native Region backend binder coverage mismatch: "
            f"missing={missing_backend_region_binders}, "
            f"unexpected={unexpected_backend_region_binders}"
        )
    provided = {
        binding.source
        for region in compilation.decision.execution_regions
        for binding in region.input_bindings
        if binding.kind
        in (
            RegionInputKind.PROVIDED,
            RegionInputKind.STAGED,
            RegionInputKind.SAMPLER_REQUEST_IDS,
            RegionInputKind.SAMPLER_SAMPLE_IDXS,
            RegionInputKind.SAMPLER_LOGITS,
            RegionInputKind.SAMPLED_TOKENS,
        )
    }
    sinks = {
        binding.target
        for region in compilation.decision.execution_regions
        for binding in region.output_bindings
        if binding.kind in (
            RegionOutputKind.SINK,
            RegionOutputKind.HOST,
            RegionOutputKind.SAMPLER,
        )
    }
    effects = tuple(
        effect
        for region in compilation.decision.execution_regions
        for effect in region.effects
    )
    state_versions = StateVersionRegistry(
        compilation.graph.states,
        bindings.state_versions,
        effects,
    )
    missing_provided = sorted(provided.difference(bindings.ports.inputs))
    if missing_provided:
        raise RuntimeError(
            "native Region missing provided input binding(s): "
            + ", ".join(missing_provided)
        )
    missing_sinks = sorted(sinks.difference(bindings.ports.outputs))
    if missing_sinks:
        raise RuntimeError(
            "native Region missing output sink binding(s): " + ", ".join(missing_sinks)
        )
    unexpected_provided = sorted(set(bindings.ports.inputs) - provided)
    unexpected_sinks = sorted(set(bindings.ports.outputs) - sinks)
    if unexpected_provided or unexpected_sinks:
        raise RuntimeError(
            "native Region runtime binding coverage mismatch: "
            f"unexpected_provided={unexpected_provided}, unexpected_sinks={unexpected_sinks}"
        )
    return state_versions


def _plan_resources(
    compilation: "WorkloadCompilation",
    *,
    graphs: tuple[RuntimeGraph, ...] = (),
) -> PlanResources:
    physical = compilation.physical
    if physical is None:
        raise RuntimeError("native compiled workload has no materialized physical plan")
    if physical.buffer_buckets:
        raise RuntimeError("native Region buffer registry is not installed")
    unsupported = tuple(
        sorted(
            {
                node.stream.value
                for node in physical.nodes
                if node.stream
                not in (StreamClass.DEFAULT, StreamClass.SIDE_FORWARD)
            }
        )
    )
    if unsupported:
        raise RuntimeError(
            f"native Region plan selected an unbound stream class: {unsupported}"
        )
    side_classes = tuple(
        sorted(
            {
                node.stream
                for node in physical.nodes
                if node.stream is not StreamClass.DEFAULT
            }
        )
    )
    used_classes = tuple(
        sorted({node.stream for node in physical.nodes})
    )
    cuda = torch.cuda.is_available()
    # Without CUDA there is no stream to overlap on; serial tokens keep the
    # signed order, which is trivially correct for synchronous host execution.
    real_side_streams = cuda and bool(side_classes)
    streams = [
        RuntimeStream(
            stream_class,
            (
                # With real side streams the fences between Regions are real
                # CUDA events, and recording one requires a real stream: the
                # main path is the ambient current stream.
                torch.cuda.current_stream()
                if real_side_streams and stream_class is StreamClass.DEFAULT
                else (
                    _SerialStream()
                    if stream_class is StreamClass.DEFAULT or not cuda
                    else torch.cuda.Stream()
                )
            ),
            retire_fence=Fence(
                FenceKey(
                    "workload",
                    f"retire-epoch-{physical.plan_epoch}-{stream_class.value}",
                ),
                _ImmediateEvent(),
            ),
        )
        for stream_class in used_classes
    ]
    event_keys = tuple(
        dependency.fence
        for node in physical.nodes
        for dependency in node.fences
    )
    if len(set(event_keys)) != len(event_keys):
        raise RuntimeError("native Region plan contains duplicate fence identities")
    return PlanResources(
        plan=physical,
        streams=tuple(streams),
        events=tuple(
            RuntimeEvent(
                key,
                torch.cuda.Event() if real_side_streams else _ImmediateEvent(),
            )
            for key in event_keys
        ),
        graphs=graphs,
    )


def _bind_physical_region_nodes(
    compilation: "WorkloadCompilation",
) -> tuple[tuple[PhysicalNodePlan, ExecutionRegion, str, ArtifactKey], ...]:
    """Bind physical nodes to signed Regions by stable node identity.

    The physical plan is a mechanical projection of a signed decision.  Its
    tuple order controls only the already-approved executor order; it must not
    decide which callable artifact implements a Region.
    """

    from tllm.schedule.physical import (
        node_key_for_region,
        artifact_key_for_region,
        executor_key_for_region,
    )
    physical = compilation.physical
    if physical is None:
        raise RuntimeError("native compiled workload has no materialized physical plan")
    if (
        not physical.schedule_signature
        or physical.schedule_signature != compilation.decision.signature
    ):
        raise RuntimeError(
            "physical Region schedule signature differs from the signed decision"
        )
    regions = compilation.decision.execution_regions
    expected_by_node: dict[NodeKey, ExecutionRegion] = {}
    for region in regions:
        node_key = node_key_for_region(compilation.graph, region.region_id)
        if node_key in expected_by_node:
            raise RuntimeError(
                f"duplicate signed Region node identity {node_key!s}"
            )
        expected_by_node[node_key] = region

    physical_by_node: dict[NodeKey, PhysicalNodePlan] = {}
    for node in physical.nodes:
        if node.node_key in physical_by_node:
            raise RuntimeError(
                f"duplicate physical Region node identity {node.node_key!s}"
            )
        physical_by_node[node.node_key] = node

    missing = tuple(sorted(set(expected_by_node) - set(physical_by_node)))
    unexpected = tuple(sorted(set(physical_by_node) - set(expected_by_node)))
    if missing or unexpected:
        raise RuntimeError(
            "physical Region identity coverage mismatch: "
            f"missing={[str(node) for node in missing]}, "
            f"unexpected={[str(node) for node in unexpected]}"
        )

    placement_by_node: dict[NodeKey, str] = {}
    for placement in physical.anchor_placements:
        if placement.node_key in placement_by_node:
            raise RuntimeError(
                f"duplicate physical Region anchor identity {placement.node_key!s}"
            )
        placement_by_node[placement.node_key] = placement.after_anchor
    missing_placements = tuple(
        sorted(set(physical_by_node) - set(placement_by_node))
    )
    unexpected_placements = tuple(
        sorted(set(placement_by_node) - set(physical_by_node))
    )
    if missing_placements or unexpected_placements:
        raise RuntimeError(
            "physical Region anchor coverage mismatch: "
            f"missing={[str(node) for node in missing_placements]}, "
            f"unexpected={[str(node) for node in unexpected_placements]}"
        )

    bound: list[tuple[PhysicalNodePlan, ExecutionRegion, str, ArtifactKey]] = []
    for node in physical.nodes:
        region = expected_by_node[node.node_key]
        if node.executor != executor_key_for_region(region):
            raise RuntimeError(
                f"physical node for {region.region_id!r} has a mismatched executor identity"
            )
        if node.stream != StreamClass(region.stream.value):
            raise RuntimeError(
                f"physical node for {region.region_id!r} stream differs from the signed decision"
            )
        if node.capture_policy != CapturePolicy(region.capture_mode.value):
            raise RuntimeError(
                f"physical node for {region.region_id!r} capture policy differs from the signed decision"
            )
        if node.fallback_policy != FallbackPolicy(region.fallback_policy.value):
            raise RuntimeError(
                f"physical node for {region.region_id!r} fallback policy differs from the signed decision"
            )
        artifact_key = node.artifact_key
        if (
            artifact_key is None
            or artifact_key != artifact_key_for_region(region)
        ):
            raise RuntimeError(
                f"physical node for {region.region_id!r} artifact key differs from the signed decision"
            )
        anchor = placement_by_node[node.node_key]
        if anchor != region.placement_anchor:
            raise RuntimeError("physical Region anchor differs from the signed decision")
        bound.append((node, region, anchor, artifact_key))
    return tuple(bound)


def _native_workload_execution(
    compilation: "WorkloadCompilation",
    *,
    runtime: "ResidualRuntimeState",
    consumers: Sequence[object],
    bindings: WorkloadExecutionBindings,
) -> NativeWorkloadExecution:
    from tllm.workload.lowering import lower_schedule_graph
    from tllm.workload.region_compiler import (
        RegionArtifactRegistry,
        materialize_region_artifacts,
    )
    from tllm.runtime.workload_plugins import (
        residual_resource_manifest_from_paths,
        static_backend_binding_manifest_wire,
    )
    from tllm.schedule.contracts import canonical_json
    from tllm import _core

    reconstructed_graph = lower_schedule_graph(
        tuple(declaration.ir for declaration in bindings.declarations),
        persistent_state_bytes=compilation.graph.persistent_state_bytes,
    )
    signed_locator_keys = {
        entry.locator_key
        for entry in compilation.options.residual_resource_manifest
    }
    # Bindings may carry legacy/PATH tap aliases in addition to the native
    # declared INDEX resources. Only the signed native manifest crosses this
    # gate; later local binding validation remains responsible for rejecting
    # unrelated runtime aliases.
    reconstructed_manifest_json = canonical_json(
        tuple(
            entry
            for entry in residual_resource_manifest_from_paths(bindings.residual_paths)
            if entry.locator_key in signed_locator_keys
        )
    )
    _core.validate_local_execution_binding_wire(
        compilation.cache_key,
        compilation.graph.to_json(),
        compilation.timeline.to_json(),
        compilation.options.to_json(),
        compilation.decision.to_json(),
        1 if compilation.physical is None else compilation.physical.plan_epoch,
        reconstructed_graph.to_json(),
        reconstructed_manifest_json,
        static_backend_binding_manifest_wire(bindings.static_binding_entries),
    )
    state_versions = _local_binding_requirements(compilation, bindings)
    artifacts = materialize_region_artifacts(
        bindings.declarations,
        compilation.decision,
        compilation.options,
    )
    registry = RegionArtifactRegistry(compilation.decision, artifacts)
    materializations = lower_signed_materialization_bindings(
        compilation.graph,
        compilation.decision,
        artifacts,
    )
    hidden_bindings = tuple(
        binding
        for artifact in artifacts
        for binding in artifact.input_bindings
        if binding.kind is RegionInputKind.HIDDEN
    )
    physical = compilation.physical
    if physical is None:
        raise RuntimeError("native compiled workload has no materialized physical plan")
    bound_nodes = _bind_physical_region_nodes(compilation)

    value_store = _RegionValueStore(
        runtime=runtime,
        providers=bindings.ports.inputs,
        sinks=bindings.ports.outputs,
        artifacts=artifacts,
        materializations=materializations,
        state_versions=state_versions,
        lifecycle=bindings.lifecycle,
    )
    handlers: dict[
        NodeKey | ExecutorKey, Callable[[PhysicalExecutionContext], None]
    ] = {}
    runtime_graphs: list[RuntimeGraph] = []
    shape_specializations = _NativeShapeSpecializationQueue()
    cost_probe = RegionCostProbe()
    anchor_nodes: dict[str, list[NodeKey]] = {}
    node_by_region: dict[str, NodeKey] = {}
    for node, region, anchor, artifact_key in bound_nodes:
        artifact = registry.resolve(region.region_id)
        stream_label = region.stream.value
        if node.capture_policy is not CapturePolicy.DISABLED:
            if not region.capture_safe:
                raise RuntimeError(
                    f"physical node for {region.region_id!r} requires an unsafe capture"
                )
            def request_shape_specialization(
                context: PhysicalExecutionContext,
                inputs: Mapping[str, torch.Tensor],
                *,
                bound_artifact: "RegionCallableArtifact" = artifact,
            ) -> RegionShapeSpecializationRequest:
                request = _region_shape_specialization_request(
                    compilation,
                    bound_artifact,
                    context,
                    inputs,
                )
                shape_specializations.enqueue(request)
                return request

            runtime_graphs.append(
                RuntimeGraph(
                    artifact_key,
                    _RegionCudaGraph(
                        artifact,
                        value_store,
                        request_shape_specialization,
                        cost_probe=cost_probe,
                        stream_label=stream_label,
                    ),
                )
            )

        def execute_region(
            context: PhysicalExecutionContext,
            *,
            bound_artifact: "RegionCallableArtifact" = artifact,
            bound_stream_label: str = stream_label,
        ) -> None:
            with cost_probe.measure(
                bound_artifact.region_id, "execute", bound_stream_label
            ):
                value_store.execute(bound_artifact, context)

        handlers[node.node_key] = execute_region
        anchor_nodes.setdefault(anchor, []).append(node.node_key)
        node_by_region[region.region_id] = node.node_key

    event_nodes: dict[BackendObservableEvent, list[NodeKey]] = {}
    for advance in compilation.decision.runtime_advance_plan:
        event_nodes.setdefault(advance.backend_event, []).extend(
            node_by_region[region_id] for region_id in advance.region_ids
        )
    known_anchors = set(anchor_nodes)
    if compilation.decision.completion_anchor not in known_anchors:
        raise RuntimeError("signed Region completion anchor has no Region binding")

    from tllm.workload import WorkloadAuthority

    if any(
        declaration.authority is not WorkloadAuthority.NATIVE_REGION
        for declaration in bindings.declarations
    ):
        raise RuntimeError(
            "native Region execution cannot bind a legacy-adapter workload"
        )
    resources = _plan_resources(compilation, graphs=tuple(runtime_graphs))
    executor = PhysicalExecutor(
        snapshots=PlanSnapshotStore(resources),
        handlers=handlers,
    )
    side_streams = tuple(
        resource.stream
        for resource in resources.streams
        if resource.key is not StreamClass.DEFAULT
        and isinstance(resource.stream, torch.cuda.Stream)
    )
    frozen_anchor_nodes = MappingProxyType(
        {anchor: tuple(nodes) for anchor, nodes in anchor_nodes.items()}
    )
    frozen_event_nodes = MappingProxyType(
        {event: tuple(nodes) for event, nodes in event_nodes.items()}
    )
    return NativeWorkloadExecution(
        compilation=compilation,
        executor=executor,
        completion_anchor=compilation.decision.completion_anchor,
        _owned_consumers=(),
        _workload_instances=cast(
            tuple[WorkloadPortOwner, ...],
            tuple(declaration.instance for declaration in bindings.declarations),
        ),
        _anchor_nodes=frozen_anchor_nodes,
        _event_nodes=frozen_event_nodes,
        _value_store=value_store,
        _state_versions=state_versions,
        _artifact_report=tuple(
            (artifact.region_id, artifact.artifact_key)
            for artifact in registry.artifacts
        ),
        _sampler=bindings.sampler,
        _diagnostics_readers=bindings.diagnostics_readers,
        _lifecycle=bindings.lifecycle,
        _backend_region_binders=bindings.backend_region_binders,
        parameter_overlays=bindings.parameter_overlays,
        _requires_decode_hidden=bool(hidden_bindings),
        _shape_specializations=shape_specializations,
        _side_streams=side_streams,
        _cost_probe=cost_probe,
    )


def build_native_workload_execution(
    compilation: "WorkloadCompilation",
    *,
    runtime: "ResidualRuntimeState",
    consumers: Sequence[object],
    bindings: WorkloadExecutionBindings | None = None,
) -> NativeWorkloadExecution | None:
    """Bind only an explicit native Region contract to its physical plan."""

    if bindings is not None:
        return _native_workload_execution(
            compilation,
            runtime=runtime,
            consumers=consumers,
            bindings=bindings,
        )
    return None


__all__ = [
    "ExecutionLifecycleBinding",
    "NativeWorkloadExecution",
    "NativeWorkloadExecutionStore",
    "NoExecutionLifecycle",
    "NoStagedExecution",
    "StateVersionBinding",
    "StateVersionRegistry",
    "StagedExecution",
    "StagedExecutionState",
    "WorkloadExecutionBindings",
    "build_native_workload_execution",
]
