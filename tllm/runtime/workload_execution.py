"""Runtime binding for compiler-owned auxiliary workload plans.

This module intentionally admits only workloads whose complete physical plan
has a fixed executor binding. Unsupported declarations remain on their
migration adapter; a supported plan with a missing runtime executor fails
before the first model forward.
"""

from __future__ import annotations

import contextlib
from _thread import LockType
from collections import Counter, deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from math import isfinite
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

import torch

from tllm.contracts.packed_hidden import (
    PackedHiddenSnapshotBatch,
)
from tllm.contracts.packed_rows import (
    PackedPhaseRowBatch,
    PackedRowPlanInputs,
    PackedSamplerRowBatch,
)
from tllm.contracts.request_identity import RequestFinishedBatch
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.contracts.workload_ports import (
    PortTransactionStatus,
    RuntimeValueMarker,
    SKIPPED,
)

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

    def request_finished_port_keys(self) -> tuple[str, ...]: ...

    def stage_dense_operand_captured(
        self, batch: DenseOperandCaptureBatch
    ) -> int: ...

from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from tllm.contracts.shape_specialization import (
    RegionShapeSpecializationInput,
    RegionShapeSpecializationRequest,
)
from tllm.runtime.region_cost_probe import RegionCostProbe, RegionCostReport
from tllm.runtime.torch_adapter import (
    torch_cuda_stream,
    torch_cuda_tensor_tuple,
    torch_record_stream_owners,
)
from tllm.runtime.parameter_overlay import (
    DenseOperandCaptureBatch,
    OverlayChannel,
    OverlayVersionStore,
)
from tllm.runtime.packed_phase_row_staging import PackedPhaseRowStaging
from tllm.runtime.workload_bindings import (
    NoPerRequestStats,
    NoSamplerBinding,
    PerRequestStats,
    PerRequestStatsService,
    OptimizerTransitionService,
    RuntimePortRegistry,
    RuntimeInputBinding,
    RuntimeOutputBinding,
    RequestStateRuntimeService,
    SamplerBinding,
    SamplerRuntimeBinding,
)
from tllm.runtime.request_trainable_state import (
    IndexedRequestIdentityBatch,
    IndexedRequestIdentityRow,
    RequestIdentityBatch,
    RequestStateEffectAudit,
    RequestStateSelectionBatch,
    RequestStateVersion,
    StateEffectReceipt,
    StateEffectOperation,
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
    PhysicalBatchExecutionContext,
    PhysicalExecutor,
    PhysicalExecutionKind,
    PhysicalNodeBatchGraph,
    PhysicalStepResult,
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
from tllm.backends.protocol import BackendObservableEvent
from tllm.workload.region_compiler import compare_region_outputs
from tllm.schedule.contracts import (
    AsyncBatchCaptureMode,
    BackendRegionBinderStage,
    ExecutionRegion,
    HostCommandKind,
    HiddenLayerKind,
    RegionInputBinding,
    RegionInputKind,
    RegionEffect,
    RegionOutputBinding,
    RegionOutputKind,
    RegionHostSubmission,
    RuntimeAdvancePlan,
    RuntimeHostCommand,
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
    from tllm.schedule.coordinator import WorkloadCompilation
    from tllm.workload.frontend import WorkloadDeclaration
    from tllm.workload.region_compiler import RegionCallableArtifact
    from tllm.workload.region_compiler import TensorRegionCallable




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

    if decision.schema_version != 7:
        raise TypeError("decision must be a ScheduleDecision schema v7")

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

    __slots__ = ("_bindings", "_observed", "_lock")

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
        effects: tuple[RegionEffect, ...] = (),
    ) -> None:
        if any(state.scope == "request" for state in states) or any(
            effect.selection_binding_id for effect in effects
        ):
            raise RuntimeError(
                "request-state resources require the typed request-state audit service"
            )
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
        self._lock = Lock()

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
        effects: tuple[RegionEffect, ...],
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
                with self._lock:
                    self._observed.update(
                        ((effect.resource, key), version)
                        for key, version in snapshot.items()
                    )
        return MappingProxyType(snapshots)

    def after(
        self,
        effects: tuple[RegionEffect, ...],
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
            with self._lock:
                self._observed.update(
                    ((resource, key), version) for key, version in current.items()
                )

    def observed_version(self, resource: str, key: Hashable) -> int:
        with self._lock:
            try:
                return self._observed[(resource, key)]
            except KeyError as exc:
                raise KeyError((resource, key)) from exc

    def observed_versions(self) -> tuple[tuple[str, Hashable, int], ...]:
        """Return a stable diagnostic snapshot without exposing mutable state."""

        with self._lock:
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
    optimizer_transitions: Mapping[str, OptimizerTransitionService] = field(
        default_factory=dict[str, OptimizerTransitionService]
    )
    request_state_services: Mapping[
        str, RequestStateRuntimeService[torch.nn.Module]
    ] = field(
        default_factory=dict[str, RequestStateRuntimeService[torch.nn.Module]]
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
    packed_hidden_resources: tuple[PackedHiddenResourceBinding, ...] = ()
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
        if any(not resource for resource in self.optimizer_transitions):
            raise ValueError("optimizer-transition resource names must not be empty")
        object.__setattr__(
            self,
            "optimizer_transitions",
            MappingProxyType(dict(self.optimizer_transitions)),
        )
        if any(
            service.resource != resource
            for resource, service in self.request_state_services.items()
        ):
            raise ValueError(
                "request-state service resource names must match their registry keys"
            )
        object.__setattr__(
            self,
            "request_state_services",
            MappingProxyType(dict(self.request_state_services)),
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
        packed_locator_keys = tuple(
            binding.locator_key for binding in self.packed_hidden_resources
        )
        if tuple(sorted(packed_locator_keys)) != packed_locator_keys or len(
            set(packed_locator_keys)
        ) != len(packed_locator_keys):
            raise ValueError(
                "packed hidden resources must have sorted unique locators"
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

    def stage_request_identities(
        self, step_id: int, batch: RequestIdentityBatch
    ) -> None:
        _ = step_id
        _ = batch
        raise RuntimeError("native execution has no request-state runtime service")

    def stage_packed_rows(
        self, step_id: int, batch: PackedPhaseRowBatch
    ) -> None:
        _ = step_id
        _ = batch
        raise RuntimeError("native execution has no signed packed-row input")

    def stage_packed_hidden_snapshots(
        self, step_id: int, batch: PackedHiddenSnapshotBatch
    ) -> None:
        _ = step_id
        _ = batch
        raise RuntimeError("native execution has no signed packed-hidden input")

    def stage_packed_sampler_rows(
        self, step_id: int, batch: PackedSamplerRowBatch
    ) -> None:
        _ = step_id
        _ = batch
        raise RuntimeError("native execution has no signed packed-sampler-row input")

    def packed_sampler_rows_for_step(
        self, step_id: int
    ) -> PackedSamplerRowBatch:
        _ = step_id
        raise RuntimeError("native execution has no signed packed-sampler-row input")

    def retire_request_states(
        self, batch: RequestFinishedBatch
    ) -> tuple[StateEffectReceipt, ...]:
        _ = batch
        return ()

    @property
    def requires_request_identities(self) -> bool:
        return False

    @property
    def requires_packed_rows(self) -> bool:
        return False

    @property
    def requires_packed_sampler_rows(self) -> bool:
        return False

    def request_state_version_report(
        self,
    ) -> tuple[tuple[str, RequestStateVersion], ...]:
        return ()

    @property
    def retained_value_count(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class _NoAsyncHostLane:
    pass


@dataclass(slots=True)
class _AsyncHostSubmission:
    submission_id: str
    step_id: int
    future: Future[PhysicalStepResult]
    retire_requested: bool = False


class _AsyncHostLane:
    __slots__ = ("executor", "active")

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tllm-async-host",
        )
        self.active: _AsyncHostSubmission | None = None


AsyncHostLane = _NoAsyncHostLane | _AsyncHostLane


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
    _event_wait_before: frozenset[BackendObservableEvent] = frozenset()
    _event_wait_after: frozenset[BackendObservableEvent] = frozenset()
    _anchor_wait_before: frozenset[str] = frozenset()
    _anchor_wait_after: frozenset[str] = frozenset()
    _event_advances: Mapping[
        BackendObservableEvent, tuple[RuntimeAdvancePlan, ...]
    ] = field(
        default_factory=dict[
            BackendObservableEvent, tuple[RuntimeAdvancePlan, ...]
        ]
    )
    _anchor_advances: Mapping[str, tuple[RuntimeAdvancePlan, ...]] = field(
        default_factory=dict[str, tuple[RuntimeAdvancePlan, ...]]
    )
    _region_nodes: Mapping[str, NodeKey] = field(
        default_factory=dict[str, NodeKey]
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
    _packed_row_staging: PackedPhaseRowStaging = field(
        default_factory=PackedPhaseRowStaging
    )
    _packed_row_staged_steps: set[int] = field(default_factory=set[int])
    _executed_nodes: set[NodeKey] = field(default_factory=set[NodeKey])
    _pending_request_finished_sources: tuple[str, ...] = ()
    _last_request_state_retirements: tuple[StateEffectReceipt, ...] = ()
    _execution_lock: LockType = field(default_factory=Lock)
    _main_stream: torch.cuda.Stream | None = None
    _side_streams: tuple[torch.cuda.Stream, ...] = ()
    _side_node_keys: frozenset[NodeKey] = frozenset()
    _side_work_pending: bool = False
    _async_host: AsyncHostLane = field(default_factory=_NoAsyncHostLane)
    _async_batch_graphs: Mapping[
        tuple[NodeKey, ...], PhysicalNodeBatchGraph
    ] = field(default_factory=dict[tuple[NodeKey, ...], PhysicalNodeBatchGraph])
    _cost_probe: RegionCostProbe = field(default_factory=RegionCostProbe)

    def _join_side_streams_locked(self) -> None:
        """Order the main stream after all side-stream work enqueued so far.

        Called only for Rust-signed before/after wait obligations and explicit
        synchronization. Side Regions may cross ordinary lifecycle events and
        step retirement; Python does not infer a consumer or scheduling edge.
        This is an async event wait, not a host synchronization.
        """
        if not self._side_work_pending:
            return
        if self._side_streams:
            if self._main_stream is None:
                raise RuntimeError(
                    "side-stream execution has no bound main CUDA stream"
                )
            for stream in self._side_streams:
                self._main_stream.wait_stream(stream)
        self._side_work_pending = False

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
        with self._execution_lock:
            if self._pending_request_finished_sources:
                raise RuntimeError(
                    "a request-finished lifecycle batch is already staged"
                )
            sources = tuple(
                source
                for workload in self._workload_instances
                for source in workload.request_finished_port_keys()
            )
            if len(sources) != len(set(sources)):
                raise RuntimeError(
                    "request-finished staged port sources must be unique"
                )
            staged = sum(
                workload.stage_request_finished(batch)
                for workload in self._workload_instances
            )
            if staged != len(sources):
                raise RuntimeError(
                    "request-finished staged port coverage differs from its typed registry"
                )
            self._pending_request_finished_sources = sources
            return staged

    def execute_staged_request_finished(self, step_id: int) -> int:
        """Execute only the signed Region closure rooted at lifecycle ports."""

        step = int(step_id)
        with self._execution_lock:
            sources = self._pending_request_finished_sources
            if not sources:
                raise RuntimeError("no request-finished lifecycle batch is staged")
            regions = self.compilation.decision.execution_regions
            source_set = frozenset(sources)
            observed_sources = frozenset(
                binding.source
                for region in regions
                for binding in region.input_bindings
                if binding.kind is RegionInputKind.STAGED
                and binding.source in source_set
            )
            if observed_sources != source_set:
                raise RuntimeError(
                    "request-finished ports do not have exact signed Region consumers"
                )
            selected = {
                region.region_id
                for region in regions
                if any(
                    binding.kind is RegionInputKind.STAGED
                    and binding.source in source_set
                    for binding in region.input_bindings
                )
            }
            changed = True
            while changed:
                changed = False
                for materialization in self.compilation.decision.materializations:
                    if (
                        materialization.producer_region in selected
                        and materialization.consumer_region not in selected
                    ):
                        selected.add(materialization.consumer_region)
                        changed = True
            region_by_id = {region.region_id: region for region in regions}
            pending_dependencies = [
                dependency
                for region_id in tuple(selected)
                for dependency in region_by_id[region_id].dependencies
            ]
            while pending_dependencies:
                dependency = pending_dependencies.pop()
                if dependency in selected:
                    continue
                selected.add(dependency)
                pending_dependencies.extend(
                    region_by_id[dependency].dependencies
                )
            selected_regions = tuple(
                region for region in regions if region.region_id in selected
            )
            if any(
                region.host_submission is not RegionHostSubmission.INLINE
                for region in selected_regions
            ):
                raise RuntimeError(
                    "request-finished Region closure requires a signed inline submission"
                )
            if self._prepared_step >= 0:
                self._finish_step_locked(self._prepared_step)
                self._drain_async_host_locked()
            self.executor.prepare_step(step)
            self._value_store.prepare_step(step)
            self._prepared_step = step
            self._executed_nodes.clear()
            nodes = tuple(
                self._region_nodes[region.region_id]
                for region in selected_regions
            )
            self._advance_nodes_locked(nodes)
            self._finish_step_locked(step)
            self._drain_async_host_locked()
            self._pending_request_finished_sources = ()
            return len(nodes)

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

        with self._execution_lock:
            self._drain_async_host_locked()
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
        with self._execution_lock:
            if self._prepared_step == step:
                return
            if self._prepared_step >= 0:
                self._finish_step_locked(self._prepared_step)
            # Retirements are a bounded observational receipt for the most
            # recently completed request batch. A new logical step starts a
            # new lifetime and must not surface stale completion evidence.
            self._last_request_state_retirements = ()
            self.executor.prepare_step(step)
            self._value_store.prepare_step(step)
            self._prepared_step = step
            self._executed_nodes.clear()

    def finish_step(self, step_id: int) -> None:
        """Retire one prepared step without releasing epoch-owned artifacts."""

        step = int(step_id)
        with self._execution_lock:
            self._finish_step_locked(step)

    def _finish_step_locked(self, step: int) -> None:
        if self._prepared_step < 0:
            return
        if self._prepared_step != step:
            raise RuntimeError("cannot retire a native workload step from another epoch")
        match self._async_host:
            case _AsyncHostLane(active=submission) if (
                submission is not None and submission.step_id == step
            ):
                submission.retire_requested = True
            case _AsyncHostLane() | _NoAsyncHostLane():
                self._retire_step_resources_locked(step)
        self._prepared_step = -1
        self._executed_nodes.clear()

    def _retire_step_resources_locked(self, step: int) -> None:
        try:
            self.executor.finish_step(step)
            self._value_store.finish_step(step)
        finally:
            if step in self._packed_row_staged_steps:
                self._packed_row_staging.finish_step(self.plan_epoch, step)
                self._packed_row_staged_steps.remove(step)
        self._cost_probe.harvest()

    def advance_anchor(self, anchor: str) -> bool:
        """Compatibility entry point for an already-signed timeline anchor."""

        anchor = str(anchor)
        advances = self._anchor_advances.get(anchor)
        if advances is not None:
            with self._execution_lock:
                advanced = False
                for advance in advances:
                    advanced = self._execute_advance_locked(advance) or advanced
                return advanced
        nodes = self._anchor_nodes.get(anchor)
        if nodes is None:
            return False
        with self._execution_lock:
            if anchor in self._anchor_wait_before:
                self._join_side_streams_locked()
            advanced = self._advance_nodes_locked(nodes)
            if anchor in self._anchor_wait_after:
                self._join_side_streams_locked()
            return advanced

    def advance_backend_event(self, backend_event: BackendObservableEvent) -> bool:
        """Advance only the Region nodes signed for one backend event."""

        advances = self._event_advances.get(backend_event)
        if advances is None:
            raise RuntimeError(
                "backend event "
                f"{backend_event.value!r} is not signed for this native workload"
            )
        with self._execution_lock:
            advanced = False
            for advance in advances:
                advanced = self._execute_advance_locked(advance) or advanced
            return advanced

    def _execute_advance_locked(self, advance: RuntimeAdvancePlan) -> bool:
        advanced = False
        side_wait_owed = advance.wait_for_prior_side_work
        for command in advance.host_commands:
            match command.kind:
                case HostCommandKind.WAIT_ASYNC:
                    self._wait_async_host_locked(command.submission_id)
                case HostCommandKind.EXECUTE_INLINE:
                    if side_wait_owed:
                        self._join_side_streams_locked()
                        side_wait_owed = False
                    advanced = (
                        self._advance_nodes_locked(
                            self._node_keys(command.region_ids)
                        )
                        or advanced
                    )
                case HostCommandKind.SUBMIT_ASYNC:
                    advanced = (
                        self._submit_async_host_locked(
                            command,
                        )
                        or advanced
                    )
        if side_wait_owed:
            self._join_side_streams_locked()
        if advance.wait_for_side_outputs:
            self._join_side_streams_locked()
        return advanced

    def _node_keys(self, region_ids: tuple[str, ...]) -> tuple[NodeKey, ...]:
        return tuple(self._region_nodes[region_id] for region_id in region_ids)

    def _advance_nodes_locked(self, nodes: tuple[NodeKey, ...]) -> bool:
        if self._prepared_step < 0:
            # vLLM kernel warmup calls sampler methods without opening a
            # logical engine step. It has no workload values to execute.
            return False
        pending = tuple(node for node in nodes if node not in self._executed_nodes)
        if not pending:
            return False
        if any(node in self._side_node_keys for node in pending):
            self._side_work_pending = True
        self.executor.execute_nodes(self._prepared_step, pending)
        self._executed_nodes.update(pending)
        return True

    def _submit_async_host_locked(
        self,
        command: RuntimeHostCommand,
    ) -> bool:
        submission_id = command.submission_id
        nodes = self._node_keys(command.region_ids)
        if self._prepared_step < 0:
            return False
        pending = tuple(node for node in nodes if node not in self._executed_nodes)
        if not pending:
            return False
        match self._async_host:
            case _AsyncHostLane(active=None) as lane:
                main_snapshot: torch.cuda.Event | None = None
                if self._main_stream is not None and self._side_streams:
                    main_snapshot = torch.cuda.Event()
                    main_snapshot.record(self._main_stream)
                match command.async_capture_mode:
                    case AsyncBatchCaptureMode.DISABLED:
                        batch_graph = None
                    case AsyncBatchCaptureMode.OPTIMIZER_TRANSITION:
                        try:
                            batch_graph = self._async_batch_graphs[pending]
                        except KeyError as exc:
                            raise RuntimeError(
                                "signed optimizer batch has no bound CUDA artifact"
                            ) from exc
                future = lane.executor.submit(
                    self.executor.execute_nodes,
                    self._prepared_step,
                    pending,
                    main_snapshot=main_snapshot,
                    batch_graph=batch_graph,
                )
                lane.active = _AsyncHostSubmission(
                    submission_id=submission_id,
                    step_id=self._prepared_step,
                    future=future,
                )
            case _AsyncHostLane():
                raise RuntimeError(
                    "async host lane capacity was not released by signed wait_async"
                )
            case _NoAsyncHostLane():
                raise RuntimeError(
                    "signed submit_async has no bound async host lane"
                )
        if any(node in self._side_node_keys for node in pending):
            self._side_work_pending = True
        self._executed_nodes.update(pending)
        return True

    def _wait_async_host_locked(self, submission_id: str) -> None:
        match self._async_host:
            case _AsyncHostLane(active=None):
                return
            case _AsyncHostLane(active=submission) as lane if (
                submission is not None
                and submission.submission_id == submission_id
            ):
                try:
                    submission.future.result()
                except Exception:
                    lane.active = None
                    if submission.retire_requested:
                        self._retire_step_resources_locked(submission.step_id)
                    raise
                lane.active = None
                self._join_side_streams_locked()
                if submission.retire_requested:
                    self._retire_step_resources_locked(submission.step_id)
            case _AsyncHostLane():
                raise RuntimeError(
                    "signed wait_async does not match the active submission"
                )
            case _NoAsyncHostLane():
                raise RuntimeError(
                    "signed wait_async has no bound async host lane"
                )

    def _drain_async_host_locked(self) -> None:
        match self._async_host:
            case _AsyncHostLane(active=submission) if submission is not None:
                self._wait_async_host_locked(submission.submission_id)
            case _AsyncHostLane() | _NoAsyncHostLane():
                pass

    def close(self) -> None:
        with self._execution_lock:
            try:
                self._drain_async_host_locked()
            finally:
                self._join_side_streams_locked()
                if self._main_stream is not None:
                    self._main_stream.synchronize()
                if self._prepared_step >= 0:
                    self._finish_step_locked(self._prepared_step)
                match self._async_host:
                    case _AsyncHostLane() as lane:
                        lane.executor.shutdown(wait=True)
                    case _NoAsyncHostLane():
                        pass

    @property
    def retained_value_count(self) -> int:
        with self._execution_lock:
            return self._value_store.retained_value_count

    @property
    def state_versions(self) -> StateVersionRegistry:
        with self._execution_lock:
            self._drain_async_host_locked()
            return self._state_versions

    def state_version_report(self) -> tuple[tuple[str, object, int], ...]:
        """Project observed versions onto the closed JSON control-plane key set."""

        with self._execution_lock:
            self._drain_async_host_locked()
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

    def request_state_version_report(
        self,
    ) -> tuple[tuple[str, RequestStateVersion], ...]:
        """Read request-scoped versions without weakening their typed identity."""

        with self._execution_lock:
            self._drain_async_host_locked()
            return self._value_store.request_state_version_report()

    def request_state_retirement_report(self) -> tuple[StateEffectReceipt, ...]:
        """Return the latest immutable request-state cleanup receipt tuple.

        This is observational evidence only. It is populated at the verified
        native request-finish safe point and cleared when the next logical
        step starts, so it cannot accumulate historical request identities.
        """

        with self._execution_lock:
            self._drain_async_host_locked()
            return self._last_request_state_retirements

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

    def stage_sampler_identity(
        self,
        step_id: int,
        request_ids: tuple[str, ...],
        sample_indices: tuple[int, ...],
        sample_index_tensor: torch.Tensor | None,
    ) -> None:
        match self._sampler:
            case SamplerRuntimeBinding(stage_identity=stage_identity):
                stage_identity(
                    step_id,
                    request_ids,
                    sample_indices,
                    sample_index_tensor,
                )
            case NoSamplerBinding():
                pass

    def stage_request_identities(
        self,
        step_id: int,
        batch: RequestIdentityBatch,
    ) -> None:
        """Stage exact request identities before a signed selection Region runs.

        This is intentionally a direct typed API. Backend hook wiring is a
        later integration concern; no untyped provider is consulted here.
        """

        step = int(step_id)
        with self._execution_lock:
            if self._prepared_step != step:
                raise RuntimeError(
                    "request identities must be staged for the prepared native step"
                )
            self._value_store.stage_request_identities(step, batch)

    def stage_packed_rows(self, batch: PackedPhaseRowBatch) -> None:
        """Stage the complete signed row provenance for the prepared step."""

        with self._execution_lock:
            if batch.plan_epoch != self.plan_epoch:
                raise RuntimeError(
                    "packed rows belong to another native execution epoch"
                )
            if self._prepared_step != batch.step_id:
                raise RuntimeError(
                    "packed rows must be staged for the prepared native step"
                )
            self._value_store.stage_packed_rows(batch.step_id, batch)

    def stage_packed_row_plan(
        self,
        step_id: int,
        columns: PackedRowPlanInputs,
        prompt_indices: Mapping[str, int],
        sample_indices: Mapping[str, int],
    ) -> PackedPhaseRowBatch:
        """Expand and stage one prepared packed-row plan under this epoch."""

        with self._execution_lock:
            if not self._value_store.requires_packed_rows:
                raise RuntimeError("native execution has no signed packed-row input")
            if self._prepared_step != step_id:
                raise RuntimeError(
                    "packed rows must be staged for the prepared native step"
                )
            batch = self._packed_row_staging.stage(
                self.plan_epoch,
                step_id,
                columns,
                prompt_indices=prompt_indices,
                sample_indices=sample_indices,
            )
            previously_staged = step_id in self._packed_row_staged_steps
            try:
                self._value_store.stage_packed_rows(step_id, batch)
            except Exception:
                if not previously_staged:
                    self._packed_row_staging.finish_step(self.plan_epoch, step_id)
                raise
            self._packed_row_staged_steps.add(step_id)
            return batch

    def stage_packed_sampler_rows(
        self, batch: PackedSamplerRowBatch
    ) -> None:
        """Stage exact sampler-to-packed links for the prepared native step."""

        with self._execution_lock:
            if batch.plan_epoch != self.plan_epoch:
                raise RuntimeError(
                    "packed sampler rows belong to another native execution epoch"
                )
            if self._prepared_step != batch.step_id:
                raise RuntimeError(
                    "packed sampler rows must be staged for the prepared native step"
                )
            self._value_store.stage_packed_sampler_rows(
                batch.step_id, batch
            )

    def packed_sampler_rows_for_step(
        self, step_id: int
    ) -> PackedSamplerRowBatch:
        """Return explicit sampler links retained for the prepared step."""

        with self._execution_lock:
            if self._prepared_step != step_id:
                raise RuntimeError(
                    "packed sampler rows require the prepared native step"
                )
            return self._value_store.packed_sampler_rows_for_step(step_id)

    def packed_rows_for_step(self, step_id: int) -> PackedPhaseRowBatch:
        """Return the canonical packed rows retained for one active step."""

        with self._execution_lock:
            if step_id not in self._packed_row_staged_steps:
                raise RuntimeError("packed rows were not staged for this native step")
            return self._packed_row_staging.batch_for_step(
                self.plan_epoch, step_id
            )

    def stage_packed_hidden_snapshots(
        self, batch: PackedHiddenSnapshotBatch
    ) -> None:
        """Stage owned eager hidden snapshots for the exact packed rows."""

        with self._execution_lock:
            if batch.plan_epoch != self.plan_epoch:
                raise RuntimeError(
                    "packed hidden snapshots belong to another native execution epoch"
                )
            if self._prepared_step != batch.step_id:
                raise RuntimeError(
                    "packed hidden snapshots must be staged for the prepared native step"
                )
            self._value_store.stage_packed_hidden_snapshots(
                batch.step_id, batch
            )

    @property
    def requires_request_identities(self) -> bool:
        return self._value_store.requires_request_identities

    @property
    def requires_packed_rows(self) -> bool:
        return self._value_store.requires_packed_rows

    @property
    def requires_packed_sampler_rows(self) -> bool:
        return self._value_store.requires_packed_sampler_rows

    @property
    def packed_hidden_resources(
        self,
    ) -> tuple[PackedHiddenResourceBinding, ...]:
        return self.compilation.options.packed_hidden_resource_manifest

    @property
    def requires_packed_hidden(self) -> bool:
        return bool(self.packed_hidden_resources)

    def retire_request_states(
        self, batch: RequestFinishedBatch
    ) -> tuple[StateEffectReceipt, ...]:
        """Retire all typed request resources at a verified execution safe point."""

        with self._execution_lock:
            self._drain_async_host_locked()
            self._join_side_streams_locked()
            if self._prepared_step >= 0:
                raise RuntimeError(
                    "request-state retirement requires the prepared native step to retire"
                )
            receipts = self._value_store.retire_request_states(batch)
            if any(
                receipt.operation is not StateEffectOperation.RETIRE
                for receipt in receipts
            ):
                raise RuntimeError(
                    "request-state retirement returned a non-retirement receipt"
                )
            self._last_request_state_retirements = receipts
            return receipts

    def read_resource_diagnostics(self) -> Mapping[str, Mapping[str, int | float]]:
        """Return a validated, JSON-safe snapshot of local workload resources.

        Resource diagnostics deliberately remain outside the signed compilation:
        they describe the active worker instance only and cannot influence
        scheduling, artifact identity, or cache selection.
        """

        with self._execution_lock:
            self._drain_async_host_locked()
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
                        raise ValueError(
                            "native diagnostic float values must be finite"
                        )
                    normalized[name] = value
                result[workload_id] = MappingProxyType(
                    dict(sorted(normalized.items()))
                )
            return MappingProxyType(dict(sorted(result.items())))

    def read_and_reset_stats(self, *, sync: bool = True) -> "TrainingStats":
        with self._execution_lock:
            self._drain_async_host_locked()
            self._join_side_streams_locked()
            execution_stream = (
                contextlib.nullcontext()
                if self._main_stream is None
                else torch.cuda.stream(self._main_stream)
            )
            with execution_stream:
                return self._lifecycle.read_and_reset_stats(bool(sync))

    def read_and_reset_per_request_stats(
        self, *, sync: bool = True
    ) -> Mapping[int, "TrainingStats"]:
        with self._execution_lock:
            self._drain_async_host_locked()
            self._join_side_streams_locked()
            execution_stream = (
                contextlib.nullcontext()
                if self._main_stream is None
                else torch.cuda.stream(self._main_stream)
            )
            with execution_stream:
                return self._lifecycle.read_and_reset_per_request_stats(
                    bool(sync)
                )

    def set_enabled(self, enabled: bool) -> None:
        _ = enabled
        raise RuntimeError(
            "native workload execution is immutable; request a coordinator-owned "
            "cold epoch transition instead"
        )

    def synchronize(self) -> None:
        with self._execution_lock:
            self._drain_async_host_locked()
            self._join_side_streams_locked()
            execution_stream = (
                contextlib.nullcontext()
                if self._main_stream is None
                else torch.cuda.stream(self._main_stream)
            )
            with execution_stream:
                self._lifecycle.synchronize()
            if self._main_stream is not None:
                self._main_stream.synchronize()

    def read_workgraph_runtime_stats(self) -> "WorkGraphRuntimeStats":
        with self._execution_lock:
            self._drain_async_host_locked()
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

@dataclass(frozen=True, slots=True)
class _RegionStateAudit:
    legacy_before: Mapping[str, Mapping[Hashable, int]]
    request_before: tuple[RequestStateEffectAudit, ...]


def _request_state_effect_operation(
    effect: RegionEffect,
) -> StateEffectOperation:
    """Mechanically lower the signed effect kind at the runtime boundary."""

    if effect.kind is ScheduleEffectKind.STATE_READ:
        return StateEffectOperation.READ
    if effect.kind is ScheduleEffectKind.STATE_TRANSITION:
        return StateEffectOperation.TRANSITION
    raise RuntimeError(
        "request-state runtime supports only strict read and transition effects"
    )


@dataclass(frozen=True, slots=True)
class _NoPackedRows:
    pass


@dataclass(frozen=True, slots=True)
class _NoPackedHiddenSnapshots:
    pass


@dataclass(frozen=True, slots=True)
class _NoPackedSamplerRows:
    pass


_PackedRowsFrameValue = _NoPackedRows | PackedPhaseRowBatch
_PackedHiddenFrameValue = _NoPackedHiddenSnapshots | PackedHiddenSnapshotBatch
_PackedSamplerRowsFrameValue = _NoPackedSamplerRows | PackedSamplerRowBatch


@dataclass(slots=True)
class _RegionStepFrame:
    values: dict[str, object] = field(default_factory=dict[str, object])
    request_selections: dict[
        str, RequestStateSelectionBatch | RuntimeValueMarker
    ] = field(
        default_factory=dict[
            str, RequestStateSelectionBatch | RuntimeValueMarker
        ]
    )
    packed_rows: _PackedRowsFrameValue = field(default_factory=_NoPackedRows)
    packed_hidden: _PackedHiddenFrameValue = field(
        default_factory=_NoPackedHiddenSnapshots
    )
    packed_sampler_rows: _PackedSamplerRowsFrameValue = field(
        default_factory=_NoPackedSamplerRows
    )
    lock: LockType = field(default_factory=Lock)


class _RegionValueStore:
    """Step-scoped binding and lifetime owner for Region boundary values."""

    __slots__ = (
        "_runtime",
        "_providers",
        "_sinks",
        "_hidden_inputs",
        "_packed_hidden_resources",
        "_packed_hidden_resources_by_binding",
        "_ordinary_request_state_resources",
        "_packed_request_state_resources",
        "_packed_request_state_phases",
        "_requires_packed_rows",
        "_requires_packed_sampler_rows",
        "_materializations",
        "_state_versions",
        "_request_state_services",
        "_lifecycle",
        "_frames",
        "_frames_lock",
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
        request_state_services: Mapping[
            str, RequestStateRuntimeService[torch.nn.Module]
        ] = MappingProxyType({}),
        packed_hidden_resources: tuple[
            PackedHiddenResourceBinding, ...
        ] = (),
    ) -> None:
        self._runtime = runtime
        self._state_versions = state_versions
        self._request_state_services = MappingProxyType(dict(request_state_services))
        self._lifecycle = lifecycle
        self._providers = providers
        self._sinks = sinks
        self._materializations = materializations
        artifact_inputs = tuple(
            binding
            for artifact in artifacts
            for binding in artifact.input_bindings
        )
        self._hidden_inputs = MappingProxyType(
            {
                binding.binding_id: binding
                for binding in artifact_inputs
                if binding.kind is RegionInputKind.HIDDEN
            }
        )
        packed_hidden_inputs = tuple(
            binding
            for binding in artifact_inputs
            if binding.kind is RegionInputKind.PACKED_HIDDEN
        )
        packed_hidden_by_locator = {
            resource.locator_key: resource
            for resource in packed_hidden_resources
        }
        required_packed_hidden = {
            (
                binding.layer,
                binding.layer_kind.value,
                binding.site.value,
            )
            for binding in packed_hidden_inputs
        }
        if required_packed_hidden != set(packed_hidden_by_locator):
            raise RuntimeError(
                "local packed-hidden resources differ from signed Region inputs"
            )
        self._packed_hidden_resources = packed_hidden_resources
        self._packed_hidden_resources_by_binding = MappingProxyType(
            {
                binding.binding_id: packed_hidden_by_locator[
                    (
                        binding.layer,
                        binding.layer_kind.value,
                        binding.site.value,
                    )
                ]
                for binding in packed_hidden_inputs
            }
        )
        packed_request_state_phases: dict[str, HiddenPhase] = {}
        for binding in artifact_inputs:
            if (
                binding.kind is not RegionInputKind.STATE_SELECTION
                or not binding.packed_rows_binding_id
            ):
                continue
            if binding.phase not in (
                HiddenPhase.PREFILL,
                HiddenPhase.DECODE,
            ):
                raise RuntimeError(
                    "packed request-state selection has no signed row phase"
                )
            if (
                binding.source in packed_request_state_phases
                and packed_request_state_phases[binding.source]
                is not binding.phase
            ):
                raise RuntimeError(
                    "one packed request-state resource cannot mix row phases"
                )
            packed_request_state_phases[binding.source] = binding.phase
        packed_request_state_resources = set(packed_request_state_phases)
        ordinary_request_state_resources = {
            binding.source
            for binding in artifact_inputs
            if binding.kind is RegionInputKind.STATE_SELECTION
            and not binding.packed_rows_binding_id
        }
        mixed_request_state_resources = ordinary_request_state_resources.intersection(
            packed_request_state_resources
        )
        if mixed_request_state_resources:
            raise RuntimeError(
                "one request-state resource cannot mix ordinary and packed identity provenance"
            )
        ordinary_request_state_resources.update(
            set(self._request_state_services).difference(
                packed_request_state_resources
            )
        )
        self._ordinary_request_state_resources = tuple(
            sorted(ordinary_request_state_resources)
        )
        self._packed_request_state_resources = tuple(
            sorted(packed_request_state_resources)
        )
        self._packed_request_state_phases = MappingProxyType(
            dict(sorted(packed_request_state_phases.items()))
        )
        self._requires_packed_rows = any(
            binding.kind is RegionInputKind.PACKED_REQUEST_ROWS
            for binding in artifact_inputs
        )
        self._requires_packed_sampler_rows = any(
            binding.kind is RegionInputKind.PACKED_SAMPLER_ROWS
            for binding in artifact_inputs
        )
        self._frames: dict[int, _RegionStepFrame] = {}
        self._frames_lock = Lock()

    @property
    def retained_value_count(self) -> int:
        with self._frames_lock:
            frames = tuple(self._frames.values())
        retained = 0
        for frame in frames:
            with frame.lock:
                retained += sum(
                    value is not SKIPPED for value in frame.values.values()
                )
        return retained

    @property
    def requires_request_identities(self) -> bool:
        return bool(self._ordinary_request_state_resources)

    @property
    def requires_packed_rows(self) -> bool:
        return self._requires_packed_rows

    @property
    def requires_packed_sampler_rows(self) -> bool:
        return self._requires_packed_sampler_rows

    def request_state_version_report(
        self,
    ) -> tuple[tuple[str, RequestStateVersion], ...]:
        return tuple(
            (resource, version)
            for resource, service in self._request_state_services.items()
            for version in service.versions()
        )

    def prepare_step(self, step_id: int) -> None:
        with self._frames_lock:
            if step_id in self._frames:
                raise RuntimeError("Region value frame is already prepared")
            self._frames[step_id] = _RegionStepFrame()

    def finish_step(self, step_id: int) -> None:
        with self._frames_lock:
            frame = self._frames.pop(step_id, None)
        if frame is None:
            return
        # The physical executor has retired the step before this path runs.
        # That is the only safe point at which the service may release a
        # selection lease. CUDA remains rejected at acquisition until a real
        # completion-stream fence is wired through this boundary.
        for service in self._request_state_services.values():
            service.finish_step(step_id)

    def stage_request_identities(
        self, step_id: int, batch: RequestIdentityBatch
    ) -> None:
        """Stage ordinary backend identities for non-packed selections."""

        frame = self._frame(step_id)
        with frame.lock:
            for resource in self._ordinary_request_state_resources:
                self._request_state_services[resource].stage_identities(
                    step_id, batch
                )

    def stage_packed_rows(
        self, step_id: int, batch: PackedPhaseRowBatch
    ) -> None:
        """Bind one complete packed-row batch and its exact state identities."""

        if not self._requires_packed_rows:
            raise RuntimeError("native execution has no signed packed-row input")
        if batch.step_id != step_id:
            raise RuntimeError("packed row batch belongs to another step")
        frame = self._frame(step_id)
        with frame.lock:
            match frame.packed_rows:
                case _NoPackedRows():
                    frame.packed_rows = batch
                case PackedPhaseRowBatch() as staged if staged == batch:
                    return
                case PackedPhaseRowBatch():
                    raise RuntimeError(
                        "packed row batch differs from the staged step"
                    )
            phases = tuple(
                dict.fromkeys(self._packed_request_state_phases.values())
            )
            indexed_by_phase = {
                phase: IndexedRequestIdentityBatch(
                    tuple(
                        IndexedRequestIdentityRow(
                            row.packed_row_index,
                            row.identity,
                        )
                        for row in batch.rows
                        if row.phase is phase
                    )
                )
                for phase in phases
            }
            for resource, phase in self._packed_request_state_phases.items():
                self._request_state_services[
                    resource
                ].stage_indexed_identities(
                    step_id,
                    indexed_by_phase[phase],
                )

    def stage_packed_hidden_snapshots(
        self, step_id: int, batch: PackedHiddenSnapshotBatch
    ) -> None:
        """Bind two owned hidden tensors to the already-staged packed rows."""

        if not self._packed_hidden_resources_by_binding:
            raise RuntimeError("native execution has no signed packed-hidden input")
        if batch.step_id != step_id:
            raise RuntimeError("packed hidden snapshots belong to another step")
        frame = self._frame(step_id)
        with frame.lock:
            match frame.packed_rows:
                case PackedPhaseRowBatch() as rows if rows == batch.row_batch:
                    pass
                case PackedPhaseRowBatch():
                    raise RuntimeError(
                        "packed hidden snapshots differ from the staged rows"
                    )
                case _NoPackedRows():
                    raise RuntimeError(
                        "packed rows must be staged before hidden snapshots"
                    )
            if tuple(
                snapshot.resource for snapshot in batch.snapshots
            ) != self._packed_hidden_resources:
                raise RuntimeError(
                    "packed hidden snapshots differ from signed local resources"
                )
            match frame.packed_hidden:
                case _NoPackedHiddenSnapshots():
                    frame.packed_hidden = batch
                case PackedHiddenSnapshotBatch() as staged if staged is batch:
                    return
                case PackedHiddenSnapshotBatch():
                    raise RuntimeError(
                        "packed hidden snapshots are already staged"
                    )

    def stage_packed_sampler_rows(
        self, step_id: int, batch: PackedSamplerRowBatch
    ) -> None:
        """Bind one exact sampler-row link batch to staged packed rows."""

        if not self._requires_packed_sampler_rows:
            raise RuntimeError(
                "native execution has no signed packed-sampler-row input"
            )
        if batch.step_id != step_id:
            raise RuntimeError("packed sampler rows belong to another step")
        frame = self._frame(step_id)
        with frame.lock:
            match frame.packed_rows:
                case PackedPhaseRowBatch() as rows if rows is batch.row_batch:
                    pass
                case PackedPhaseRowBatch():
                    raise RuntimeError(
                        "packed sampler rows differ from the staged rows"
                    )
                case _NoPackedRows():
                    raise RuntimeError(
                        "packed rows must be staged before sampler rows"
                    )
            match frame.packed_sampler_rows:
                case _NoPackedSamplerRows():
                    frame.packed_sampler_rows = batch
                case PackedSamplerRowBatch() as staged if staged is batch:
                    return
                case PackedSamplerRowBatch():
                    raise RuntimeError(
                        "packed sampler rows are already staged"
                    )

    def packed_sampler_rows_for_step(
        self, step_id: int
    ) -> PackedSamplerRowBatch:
        frame = self._frame(step_id)
        with frame.lock:
            match frame.packed_sampler_rows:
                case PackedSamplerRowBatch() as sampler_rows:
                    return sampler_rows
                case _NoPackedSamplerRows():
                    raise RuntimeError(
                        "packed sampler rows were not staged for this step"
                    )

    def retire_request_states(
        self, batch: RequestFinishedBatch
    ) -> tuple[StateEffectReceipt, ...]:
        return tuple(
            service.retire_requests(batch)
            for service in self._request_state_services.values()
        )

    def _frame(self, step_id: int) -> _RegionStepFrame:
        with self._frames_lock:
            try:
                return self._frames[step_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"Region value frame for step {step_id} is not prepared"
                ) from exc

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
        torch_record_stream_owners(tuple(inputs.values()), context.stream)
        if any(value is SKIPPED for value in inputs.values()):
            self.observe_skipped_state_effects(artifact, context)
            outputs = {key: SKIPPED for key in artifact.output_keys}
        else:
            state_before = self.begin_state_effects(artifact, context)
            outputs = artifact.execute(inputs)
            torch_record_stream_owners(
                tuple(outputs.values()), context.stream
            )
            self.finish_state_effects(artifact, context, state_before)
        self.commit_outputs(artifact, context, outputs)

    def is_enabled(self) -> bool:
        return self._lifecycle.is_enabled()

    def observe_skipped_state_effects(
        self,
        artifact: "RegionCallableArtifact",
        context: PhysicalExecutionContext,
    ) -> None:
        """Record legacy baseline versions without auditing an unrun task."""

        legacy_effects = tuple(
            effect
            for effect in artifact.effects
            if effect.resource not in self._request_state_services
        )
        self._state_versions.before(legacy_effects, context.step_id)

    def begin_state_effects(
        self,
        artifact: "RegionCallableArtifact",
        context: PhysicalExecutionContext,
    ) -> _RegionStateAudit:
        legacy_effects = tuple(
            effect
            for effect in artifact.effects
            if effect.resource not in self._request_state_services
        )
        request_effects = tuple(
            effect
            for effect in artifact.effects
            if effect.resource in self._request_state_services
        )
        frame = self._frame(context.step_id)
        with frame.lock:
            selection_bindings = {
                binding.binding_id: binding
                for binding in artifact.input_bindings
                if binding.kind is RegionInputKind.STATE_SELECTION
            }
            request_before: list[RequestStateEffectAudit] = []
            for effect in request_effects:
                selection_binding = selection_bindings.get(
                    effect.selection_binding_id
                )
                if (
                    selection_binding is None
                    or selection_binding.source != effect.resource
                ):
                    raise RuntimeError(
                        "request-state effect does not name its signed selection binding"
                    )
                operation = _request_state_effect_operation(effect)
                request_before.append(
                    self._request_state_services[effect.resource].before(
                        operation,
                        self._required_selection_for(
                            frame,
                            context.step_id,
                            effect.resource,
                            (
                                torch_cuda_stream(context.stream)
                                if self._request_state_services[
                                    effect.resource
                                ].device.type
                                == "cuda"
                                else None
                            ),
                        ),
                        context.step_id,
                    )
                )
        return _RegionStateAudit(
            legacy_before=self._state_versions.before(
                legacy_effects,
                context.step_id,
            ),
            request_before=tuple(request_before),
        )

    def finish_state_effects(
        self,
        artifact: "RegionCallableArtifact",
        context: PhysicalExecutionContext,
        before: _RegionStateAudit,
    ) -> None:
        legacy_effects = tuple(
            effect
            for effect in artifact.effects
            if effect.resource not in self._request_state_services
        )
        self._state_versions.after(
            legacy_effects,
            context.step_id,
            before.legacy_before,
        )
        for audit in before.request_before:
            service = self._request_state_services[audit.resource]
            service.after(audit)
            if service.device.type == "cuda":
                service.record_completion(
                    context.step_id,
                    audit.selection,
                    torch_cuda_stream(context.stream),
                )

    def _selection_for(
        self,
        frame: _RegionStepFrame,
        step_id: int,
        resource: str,
        stream: torch.cuda.Stream | None,
    ) -> RequestStateSelectionBatch | RuntimeValueMarker:
        try:
            return frame.request_selections[resource]
        except KeyError:
            try:
                service = self._request_state_services[resource]
            except KeyError as exc:
                raise RuntimeError(
                    f"native Region has no typed request-state service for {resource!r}"
                ) from exc
            selection = service.selection_value_for_step(step_id, stream)
            value = SKIPPED if selection is None else selection
            frame.request_selections[resource] = value
            return value

    def _required_selection_for(
        self,
        frame: _RegionStepFrame,
        step_id: int,
        resource: str,
        stream: torch.cuda.Stream | None,
    ) -> RequestStateSelectionBatch:
        selection = self._selection_for(
            frame, step_id, resource, stream
        )
        if selection is SKIPPED:
            raise RuntimeError(
                "skipped request-state selection reached effect execution"
            )
        return selection

    def _required_packed_rows(
        self,
        frame: _RegionStepFrame,
        context: PhysicalExecutionContext,
    ) -> PackedPhaseRowBatch:
        match frame.packed_rows:
            case PackedPhaseRowBatch() as rows:
                if (
                    rows.plan_epoch != context.plan_epoch
                    or rows.step_id != context.step_id
                ):
                    raise RuntimeError(
                        "packed rows differ from the executing plan epoch or step"
                    )
                return rows
            case _NoPackedRows():
                raise RuntimeError(
                    "packed rows must be staged before Region execution"
                )

    def _required_packed_hidden(
        self,
        frame: _RegionStepFrame,
        context: PhysicalExecutionContext,
        binding: RegionInputBinding,
    ) -> torch.Tensor:
        rows = self._required_packed_rows(frame, context)
        match frame.packed_hidden:
            case PackedHiddenSnapshotBatch() as snapshots:
                if snapshots.row_batch != rows:
                    raise RuntimeError(
                        "packed hidden snapshots differ from executing rows"
                    )
            case _NoPackedHiddenSnapshots():
                raise RuntimeError(
                    "packed hidden snapshots must be staged before Region execution"
                )
        resource = self._packed_hidden_resources_by_binding[
            binding.binding_id
        ]
        return snapshots.hidden_for(resource)

    def _required_packed_sampler_rows(
        self,
        frame: _RegionStepFrame,
        context: PhysicalExecutionContext,
    ) -> PackedSamplerRowBatch:
        rows = self._required_packed_rows(frame, context)
        match frame.packed_sampler_rows:
            case PackedSamplerRowBatch() as sampler_rows:
                if sampler_rows.row_batch is not rows:
                    raise RuntimeError(
                        "packed sampler rows differ from executing rows"
                    )
                return sampler_rows
            case _NoPackedSamplerRows():
                raise RuntimeError(
                    "packed sampler rows must be staged before Region execution"
                )

    def resolve_inputs(
        self,
        artifact: "RegionCallableArtifact",
        context: PhysicalExecutionContext,
    ) -> Mapping[str, object]:
        frame = self._frame(context.step_id)
        with frame.lock:
            inputs: dict[str, object] = {}
            for binding in artifact.input_bindings:
                if binding.kind is RegionInputKind.MATERIALIZED:
                    self._materializations.input_binding_for(
                        artifact.region_id, binding.binding_id
                    )
                    if binding.binding_id not in frame.values:
                        raise RuntimeError(
                            f"ExecutionRegion {artifact.region_id!r} has no materialized input {binding.binding_id!r}"
                        )
                    inputs[binding.binding_id] = frame.values[
                        binding.binding_id
                    ]
                elif binding.kind is RegionInputKind.STEP_ID:
                    inputs[binding.binding_id] = context.step_id
                elif binding.kind in (
                    RegionInputKind.PROVIDED,
                    RegionInputKind.STAGED,
                    RegionInputKind.SAMPLER_REQUEST_IDS,
                    RegionInputKind.SAMPLER_SAMPLE_IDXS,
                    RegionInputKind.SAMPLER_SAMPLE_IDX_TENSOR,
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
                elif binding.kind is RegionInputKind.PACKED_REQUEST_ROWS:
                    inputs[binding.binding_id] = self._required_packed_rows(
                        frame, context
                    )
                elif binding.kind is RegionInputKind.PACKED_HIDDEN:
                    inputs[binding.binding_id] = self._required_packed_hidden(
                        frame,
                        context,
                        binding,
                    )
                elif binding.kind is RegionInputKind.PACKED_SAMPLER_ROWS:
                    inputs[binding.binding_id] = self._required_packed_sampler_rows(
                        frame, context
                    )
                elif binding.kind is RegionInputKind.STATE_SELECTION:
                    service = self._request_state_services[binding.source]
                    inputs[binding.binding_id] = self._selection_for(
                        frame,
                        context.step_id,
                        binding.source,
                        (
                            torch_cuda_stream(context.stream)
                            if service.device.type == "cuda"
                            else None
                        ),
                    )
                else:
                    raise AssertionError(
                        f"unhandled signed Region input kind {binding.kind!r}"
                    )
            return inputs

    def commit_outputs(
        self,
        artifact: "RegionCallableArtifact",
        context: PhysicalExecutionContext,
        outputs: Mapping[str, object],
    ) -> None:
        frame = self._frame(context.step_id)
        if set(outputs) != set(artifact.output_keys):
            raise RuntimeError(
                f"ExecutionRegion {artifact.region_id!r} produced invalid output bindings"
            )
        with frame.lock:
            for binding in artifact.output_bindings:
                value = outputs[binding.binding_id]
                if binding.kind in (
                    RegionOutputKind.SINK,
                    RegionOutputKind.HOST,
                    RegionOutputKind.SAMPLER,
                ):
                    if value is not SKIPPED:
                        self._sinks[binding.target].publish(
                            value, context.step_id
                        )
                elif binding.kind is RegionOutputKind.MATERIALIZED:
                    self._materializations.output_binding_for(
                        artifact.region_id, binding.binding_id
                    )
                    frame.values[binding.binding_id] = value
                else:
                    raise AssertionError(
                        f"unhandled signed Region output kind {binding.kind!r}"
                    )

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
                frame.values.pop(key, None)


@dataclass(frozen=True, slots=True)
class _TensorRegionArtifact:
    region_id: str
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    callable: "TensorRegionCallable"

    def execute(
        self, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        if len(inputs) != len(self.input_keys):
            raise RuntimeError(
                f"Tensor Region {self.region_id!r} received the wrong input arity"
            )
        with torch.inference_mode(False), torch.enable_grad():
            return self.callable(inputs)


def _tensor_region_artifact(
    artifact: "RegionCallableArtifact",
) -> _TensorRegionArtifact:
    tensor_callable = artifact.tensor_callable
    if tensor_callable is None:
        raise RuntimeError(
            f"Region {artifact.region_id!r} has no Tensor batch artifact"
        )
    return _TensorRegionArtifact(
        region_id=artifact.region_id,
        input_keys=artifact.input_keys,
        output_keys=artifact.output_keys,
        callable=tensor_callable,
    )


class _OptimizerBatchCudaGraph(PhysicalNodeBatchGraph):
    """One fail-closed CUDA graph for a Rust-signed optimizer batch."""

    __slots__ = (
        "_artifact_key",
        "_artifacts",
        "_tensor_regions",
        "_transition",
        "_transition_input",
        "_store",
        "_graph",
        "_staging",
        "_input_specs",
        "_outputs",
        "_loss",
        "_shape_specialization_request",
        "_preflight_identity",
        "_preflight_inputs",
        "_preflight_disabled",
        "_preflight_miss",
        "_poisoned",
        "_warmed",
        "_cost_probe",
        "_stream_label",
    )

    def __init__(
        self,
        artifact_key: str,
        artifacts: tuple["RegionCallableArtifact", ...],
        transition: OptimizerTransitionService,
        store: _RegionValueStore,
        shape_specialization_request: Callable[
            [PhysicalExecutionContext, Mapping[str, torch.Tensor]],
            RegionShapeSpecializationRequest,
        ],
        cost_probe: RegionCostProbe,
        stream_label: str,
    ) -> None:
        if len(artifacts) < 2:
            raise RuntimeError(
                "optimizer batch capture requires compute and transition Regions"
            )
        transition_artifact = artifacts[-1]
        if (
            len(transition_artifact.input_keys) != 1
            or transition_artifact.output_keys
            or len(transition_artifact.effects) != 1
            or transition_artifact.effects[0].kind
            is not ScheduleEffectKind.STATE_TRANSITION
        ):
            raise RuntimeError(
                "optimizer batch transition artifact differs from its signed contract"
            )
        tensor_regions = tuple(
            _tensor_region_artifact(artifact) for artifact in artifacts[:-1]
        )
        available = set(tensor_regions[0].input_keys)
        for region in tensor_regions:
            if any(key not in available for key in region.input_keys):
                raise RuntimeError(
                    "optimizer batch Tensor Regions do not form one closed value chain"
                )
            available.update(region.output_keys)
        transition_input = transition_artifact.input_keys[0]
        if transition_input not in available:
            raise RuntimeError(
                "optimizer batch transition input is not produced by its Tensor chain"
            )
        self._artifact_key = artifact_key
        self._artifacts = artifacts
        self._tensor_regions = tensor_regions
        self._transition = transition
        self._transition_input = transition_input
        self._store = store
        self._graph: torch.cuda.CUDAGraph | None = None
        self._staging: tuple[torch.Tensor, ...] = ()
        self._input_specs: tuple[
            tuple[tuple[int, ...], torch.dtype, torch.device], ...
        ] = ()
        self._outputs: tuple[tuple[torch.Tensor, ...], ...] = ()
        self._loss: torch.Tensor | None = None
        self._shape_specialization_request = shape_specialization_request
        self._preflight_identity: tuple[int, int] | None = None
        self._preflight_inputs: tuple[torch.Tensor, ...] | None = None
        self._preflight_disabled = False
        self._preflight_miss: RegionShapeSpecializationRequest | None = None
        self._poisoned = False
        self._warmed = False
        self._cost_probe = cost_probe
        self._stream_label = stream_label

    @staticmethod
    def _context_identity(
        context: PhysicalBatchExecutionContext,
    ) -> tuple[int, int]:
        return (context.plan_epoch, context.step_id)

    def _clear_preflight(self) -> None:
        self._preflight_identity = None
        self._preflight_inputs = None
        self._preflight_disabled = False
        self._preflight_miss = None

    def _device_step(
        self, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[tuple[torch.Tensor, ...], ...], torch.Tensor]:
        first = self._tensor_regions[0]
        values = dict(zip(first.input_keys, inputs, strict=True))
        region_outputs: list[tuple[torch.Tensor, ...]] = []
        for region in self._tensor_regions:
            outputs = region.execute(
                tuple(values[key] for key in region.input_keys)
            )
            values.update(zip(region.output_keys, outputs, strict=True))
            region_outputs.append(outputs)
        loss = values[self._transition_input]
        self._transition.device_step(loss)
        return tuple(region_outputs), loss

    def _capture(
        self,
        inputs: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream,
    ) -> None:
        staging = tuple(torch.empty_like(value) for value in inputs)
        for target, source in zip(staging, inputs, strict=True):
            target.copy_(source)
        graph = torch.cuda.CUDAGraph()
        # The Rust-signed batch runs on the dedicated async host thread.
        # Thread-local capture keeps unrelated main-thread CUDA submission
        # legal while preserving fail-closed capture errors in this thread.
        with torch.cuda.graph(
            graph,
            stream=stream,
            capture_error_mode="thread_local",
        ):
            outputs, loss = self._device_step(staging)
        self._graph = graph
        self._staging = staging
        self._input_specs = tuple(
            (tuple(value.shape), value.dtype, value.device) for value in inputs
        )
        self._outputs = outputs
        self._loss = loss

    def preflight_context(self, context: PhysicalBatchExecutionContext) -> None:
        identity = self._context_identity(context)
        if self._preflight_identity == identity:
            request = self._preflight_miss
            if request is not None:
                self._clear_preflight()
                raise ShapeSpecializationMiss(request)
            return
        self._clear_preflight()
        self._preflight_identity = identity
        try:
            if self._poisoned:
                raise RuntimeError(
                    f"optimizer batch CUDA artifact {self._artifact_key!r} is poisoned"
                )
            if not self._store.is_enabled():
                self._preflight_disabled = True
                return
            first_artifact = self._artifacts[0]
            physical_context = PhysicalExecutionContext(
                step_id=context.step_id,
                plan_epoch=context.plan_epoch,
                node=context.nodes[0],
                kind=PhysicalExecutionKind.REPLAY,
                stream=context.stream,
                tensors=(),
                leases=(),
            )
            raw_inputs = self._store.resolve_inputs(
                first_artifact, physical_context
            )
            values = tuple(raw_inputs[key] for key in first_artifact.input_keys)
            if any(value is SKIPPED for value in values):
                self._preflight_disabled = True
                return
            inputs = torch_cuda_tensor_tuple(values)
            if self._graph is not None:
                specs = tuple(
                    (tuple(value.shape), value.dtype, value.device)
                    for value in inputs
                )
                if specs != self._input_specs:
                    mapping = dict(zip(first_artifact.input_keys, inputs, strict=True))
                    physical_context = PhysicalExecutionContext(
                        step_id=context.step_id,
                        plan_epoch=context.plan_epoch,
                        node=context.nodes[0],
                        kind=PhysicalExecutionKind.REPLAY,
                        stream=context.stream,
                        tensors=(),
                        leases=(),
                    )
                    request = self._shape_specialization_request(
                        physical_context, mapping
                    )
                    self._preflight_miss = request
                    raise ShapeSpecializationMiss(request)
            self._preflight_inputs = inputs
        except BaseException:
            self._clear_preflight()
            raise

    def _commit_skipped(self, context: PhysicalBatchExecutionContext) -> None:
        physical_context = PhysicalExecutionContext(
            step_id=context.step_id,
            plan_epoch=context.plan_epoch,
            node=context.nodes[0],
            kind=PhysicalExecutionKind.REPLAY,
            stream=context.stream,
            tensors=(),
            leases=(),
        )
        for artifact in self._artifacts:
            self._store.commit_outputs(
                artifact,
                physical_context,
                {key: SKIPPED for key in artifact.output_keys},
            )

    def replay_context(
        self, context: PhysicalBatchExecutionContext
    ) -> PhysicalExecutionKind:
        try:
            self.preflight_context(context)
            if self._preflight_disabled:
                self._commit_skipped(context)
                return PhysicalExecutionKind.EAGER
            inputs = self._preflight_inputs
            if inputs is None:
                raise RuntimeError("optimizer batch replay was not preflighted")
            stream = context.stream
            physical_context = PhysicalExecutionContext(
                step_id=context.step_id,
                plan_epoch=context.plan_epoch,
                node=context.nodes[0],
                kind=PhysicalExecutionKind.REPLAY,
                stream=stream,
                tensors=(),
                leases=(),
            )
            state_before = tuple(
                self._store.begin_state_effects(artifact, physical_context)
                for artifact in self._artifacts
            )
            torch_record_stream_owners(inputs, stream)
            try:
                graph = self._graph
                if not self._warmed:
                    with self._cost_probe.measure(
                        self._artifact_key,
                        "warmup",
                        self._stream_label,
                    ):
                        outputs, loss = self._device_step(inputs)
                    self._outputs = outputs
                    self._loss = loss
                    self._warmed = True
                    execution_kind = PhysicalExecutionKind.EAGER
                elif graph is None:
                    with self._cost_probe.measure_host(
                        self._artifact_key,
                        "capture",
                        self._stream_label,
                    ):
                        self._capture(inputs, stream)
                    graph = self._graph
                    if graph is None:
                        raise RuntimeError(
                            "optimizer batch capture produced no CUDA graph"
                        )
                    with self._cost_probe.measure(
                        self._artifact_key,
                        "replay",
                        self._stream_label,
                    ):
                        graph.replay()
                    execution_kind = PhysicalExecutionKind.REPLAY
                else:
                    for target, source in zip(
                        self._staging, inputs, strict=True
                    ):
                        target.copy_(source)
                    with self._cost_probe.measure(
                        self._artifact_key,
                        "replay",
                        self._stream_label,
                    ):
                        graph.replay()
                    execution_kind = PhysicalExecutionKind.REPLAY
            except BaseException:
                self._poisoned = True
                for artifact, before in zip(
                    self._artifacts[:-1], state_before[:-1], strict=True
                ):
                    self._store.finish_state_effects(
                        artifact, physical_context, before
                    )
                raise
            try:
                loss = self._loss
                if loss is None:
                    raise RuntimeError(
                        "optimizer batch capture produced no loss Tensor"
                    )
                for artifact, before in zip(
                    self._artifacts[:-1], state_before[:-1], strict=True
                ):
                    self._store.finish_state_effects(
                        artifact, physical_context, before
                    )
                self._transition.commit(loss)
                self._store.finish_state_effects(
                    self._artifacts[-1], physical_context, state_before[-1]
                )
                for artifact, outputs in zip(
                    self._artifacts[:-1], self._outputs, strict=True
                ):
                    self._store.commit_outputs(
                        artifact,
                        physical_context,
                        dict(
                            zip(
                                artifact.output_keys,
                                outputs,
                                strict=True,
                            )
                        ),
                    )
                self._store.commit_outputs(
                    self._artifacts[-1], physical_context, {}
                )
                return execution_kind
            except BaseException:
                self._poisoned = True
                raise
        finally:
            self._clear_preflight()


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
        if any(
            effect.kind is not ScheduleEffectKind.STATE_READ
            for effect in artifact.effects
        ):
            raise RuntimeError(
                f"capture-safe Region {artifact.region_id!r} may contain only state-read effects"
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
                self._clear_preflight()
                raise ShapeSpecializationMiss(request)
            return
        self._clear_preflight()
        self._preflight_identity = identity
        try:
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
        except BaseException:
            self._clear_preflight()
            raise

    def replay_context(self, context: object) -> None:
        if not isinstance(context, PhysicalExecutionContext):
            raise TypeError("Region CUDA Graph requires PhysicalExecutionContext")
        try:
            self.preflight_context(context)
            if self._preflight_disabled:
                self._store.commit_outputs(
                    self._artifact,
                    context,
                    {key: SKIPPED for key in self._artifact.output_keys},
                )
                return
            tensors = self._preflight_inputs
            if tensors is None:
                raise RuntimeError("Region CUDA Graph replay was not preflighted")
            state_before = (
                self._store.begin_state_effects(self._artifact, context)
                if self._artifact.effects
                else None
            )
            probe = self._cost_probe
            if self._graph is None:
                # Capture is one-time and host-visible; measure its wall time.
                timer = (
                    probe.measure_host(
                        self._artifact.region_id,
                        "capture",
                        self._stream_label,
                    )
                    if probe is not None
                    else contextlib.nullcontext()
                )
                with timer:
                    self._capture(tensors)
            else:
                stream = self._stream
                graph = self._graph
                assert stream is not None and graph is not None
                current = torch.cuda.current_stream(
                    next(iter(tensors.values())).device
                )
                torch_record_stream_owners(tuple(tensors.values()), stream)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    # Events recorded inside this block live on the capture stream,
                    # so the reading is the graph's own GPU span, not the fencing.
                    timer = (
                        probe.measure(
                            self._artifact.region_id,
                            "replay",
                            self._stream_label,
                        )
                        if probe is not None
                        else contextlib.nullcontext()
                    )
                    with timer:
                        for key, value in tensors.items():
                            self._staging[key].copy_(value)
                        graph.replay()
                current.wait_stream(stream)
                torch_record_stream_owners(self._outputs, current)
            outputs = dict(
                zip(self._artifact.output_keys, self._outputs, strict=True)
            )
            if state_before is not None:
                self._store.finish_state_effects(
                    self._artifact,
                    context,
                    state_before,
                )
            self._store.commit_outputs(self._artifact, context, outputs)
        finally:
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
    # vLLM capture buffers are overwritten in place on the next engine step,
    # regardless of whether Torch marks their storage as an inference tensor.
    # Native autograd therefore owns an immutable ordinary-tensor snapshot at
    # this explicit backend boundary.
    with torch.inference_mode(False):
        return selected.clone()


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
            RegionInputKind.SAMPLER_SAMPLE_IDX_TENSOR,
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
    request_resources = {
        state.resource
        for state in compilation.graph.states
        if state.scope == "request"
    }
    request_selection_resources = {
        binding.source
        for region in compilation.decision.execution_regions
        for binding in region.input_bindings
        if binding.kind is RegionInputKind.STATE_SELECTION
    }
    unknown_request_selections = sorted(
        request_selection_resources - request_resources
    )
    if unknown_request_selections:
        raise RuntimeError(
            "native Region state-selection references a non-request resource: "
            + ", ".join(unknown_request_selections)
        )
    required_request_services = request_selection_resources.union(
        effect.resource for effect in effects if effect.resource in request_resources
    )
    supplied_request_services = set(bindings.request_state_services)
    missing_request_services = sorted(
        required_request_services - supplied_request_services
    )
    unexpected_request_services = sorted(
        supplied_request_services - required_request_services
    )
    if missing_request_services or unexpected_request_services:
        raise RuntimeError(
            "native Region request-state service coverage mismatch: "
            f"missing={missing_request_services}, "
            f"unexpected={unexpected_request_services}"
        )
    legacy_request_bindings = sorted(
        set(bindings.state_versions).intersection(request_resources)
    )
    if legacy_request_bindings:
        raise RuntimeError(
            "request-state resources must not enter the legacy Hashable state "
            "registry: "
            + ", ".join(legacy_request_bindings)
        )
    state_versions = StateVersionRegistry(
        tuple(
            state
            for state in compilation.graph.states
            if state.scope != "request"
        ),
        bindings.state_versions,
        tuple(
            effect for effect in effects if effect.resource not in request_resources
        ),
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
    require_cuda_default_stream: bool = False,
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
    real_cuda_streams = cuda and (
        bool(side_classes) or require_cuda_default_stream
    )
    streams = [
        RuntimeStream(
            stream_class,
            (
                # With real side streams the fences between Regions are real
                # CUDA events, and recording one requires a real stream: the
                # main path is the ambient current stream.
                torch.cuda.current_stream()
                if real_cuda_streams and stream_class is StreamClass.DEFAULT
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
                torch.cuda.Event() if real_cuda_streams else _ImmediateEvent(),
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

    expected_node_order = tuple(expected_by_node)
    physical_node_order = tuple(node.node_key for node in physical.nodes)
    if physical_node_order != expected_node_order:
        raise RuntimeError(
            "physical Region order differs from the signed decision"
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
        canonical_json(bindings.packed_hidden_resources),
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
        request_state_services=bindings.request_state_services,
        packed_hidden_resources=bindings.packed_hidden_resources,
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

    async_batch_graphs: dict[
        tuple[NodeKey, ...], PhysicalNodeBatchGraph
    ] = {}
    signed_regions = {
        region.region_id: region
        for region in compilation.decision.execution_regions
    }
    for advance in compilation.decision.runtime_advance_plan:
        for command in advance.host_commands:
            if (
                command.async_capture_mode
                is not AsyncBatchCaptureMode.OPTIMIZER_TRANSITION
            ):
                continue
            command_artifacts = tuple(
                registry.resolve(region_id) for region_id in command.region_ids
            )
            transition_artifact = command_artifacts[-1]
            if len(transition_artifact.effects) != 1:
                raise RuntimeError(
                    "signed optimizer batch has an invalid transition effect"
                )
            resource = transition_artifact.effects[0].resource
            try:
                transition_service = bindings.optimizer_transitions[resource]
            except KeyError as exc:
                raise RuntimeError(
                    "signed optimizer batch has no typed transition service for "
                    f"state {resource!r}"
                ) from exc
            first_artifact = command_artifacts[0]

            def request_batch_shape_specialization(
                context: PhysicalExecutionContext,
                inputs: Mapping[str, torch.Tensor],
                *,
                bound_artifact: "RegionCallableArtifact" = first_artifact,
            ) -> RegionShapeSpecializationRequest:
                request = _region_shape_specialization_request(
                    compilation,
                    bound_artifact,
                    context,
                    inputs,
                )
                shape_specializations.enqueue(request)
                return request

            batch_nodes = tuple(
                node_by_region[region_id] for region_id in command.region_ids
            )
            if batch_nodes in async_batch_graphs:
                raise RuntimeError(
                    "signed optimizer batch membership is duplicated"
                )
            async_batch_graphs[batch_nodes] = _OptimizerBatchCudaGraph(
                command.async_artifact_key,
                command_artifacts,
                transition_service,
                value_store,
                request_batch_shape_specialization,
                cost_probe,
                signed_regions[command.region_ids[0]].stream.value,
            )

    event_nodes: dict[BackendObservableEvent, list[NodeKey]] = {}
    event_advances: dict[
        BackendObservableEvent, list[RuntimeAdvancePlan]
    ] = {}
    anchor_advances: dict[str, list[RuntimeAdvancePlan]] = {}
    for advance in compilation.decision.runtime_advance_plan:
        event_nodes.setdefault(advance.backend_event, []).extend(
            node_by_region[region_id] for region_id in advance.region_ids
        )
        event_advances.setdefault(advance.backend_event, []).append(advance)
        anchor_advances.setdefault(advance.anchor_id, []).append(advance)
    event_wait_before = frozenset(
        advance.backend_event
        for advance in compilation.decision.runtime_advance_plan
        if advance.wait_for_prior_side_work
    )
    event_wait_after = frozenset(
        advance.backend_event
        for advance in compilation.decision.runtime_advance_plan
        if advance.wait_for_side_outputs
    )
    anchor_wait_before = frozenset(
        advance.anchor_id
        for advance in compilation.decision.runtime_advance_plan
        if advance.wait_for_prior_side_work
    )
    anchor_wait_after = frozenset(
        advance.anchor_id
        for advance in compilation.decision.runtime_advance_plan
        if advance.wait_for_side_outputs
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
    resources = _plan_resources(
        compilation,
        graphs=tuple(runtime_graphs),
        require_cuda_default_stream=any(
            service.device.type == "cuda"
            for service in bindings.request_state_services.values()
        ),
    )
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
    main_stream = next(
        (
            resource.stream
            for resource in resources.streams
            if resource.key is StreamClass.DEFAULT
            and isinstance(resource.stream, torch.cuda.Stream)
        ),
        None,
    )
    side_node_keys = frozenset(
        node.node_key
        for node in resources.plan.nodes
        if node.stream is not StreamClass.DEFAULT
    )
    frozen_anchor_nodes = MappingProxyType(
        {anchor: tuple(nodes) for anchor, nodes in anchor_nodes.items()}
    )
    frozen_event_nodes = MappingProxyType(
        {event: tuple(nodes) for event, nodes in event_nodes.items()}
    )
    frozen_event_advances = MappingProxyType(
        {
            event: tuple(advances)
            for event, advances in event_advances.items()
        }
    )
    frozen_anchor_advances = MappingProxyType(
        {
            anchor: tuple(advances)
            for anchor, advances in anchor_advances.items()
        }
    )
    async_host: AsyncHostLane = (
        _AsyncHostLane()
        if any(
            region.host_submission is RegionHostSubmission.ASYNC_SERIAL
            for region in compilation.decision.execution_regions
        )
        else _NoAsyncHostLane()
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
        _event_wait_before=event_wait_before,
        _event_wait_after=event_wait_after,
        _anchor_wait_before=anchor_wait_before,
        _anchor_wait_after=anchor_wait_after,
        _event_advances=frozen_event_advances,
        _anchor_advances=frozen_anchor_advances,
        _region_nodes=MappingProxyType(dict(node_by_region)),
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
        _main_stream=main_stream,
        _side_streams=side_streams,
        _side_node_keys=side_node_keys,
        _async_host=async_host,
        _async_batch_graphs=MappingProxyType(async_batch_graphs),
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
