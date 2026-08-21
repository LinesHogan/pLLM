"""Deterministic structural serialization for WorkGraph descriptors.

The serializer is deliberately closed-world.  WorkGraph metadata has an
explicit schema, so accepting arbitrary mappings or objects here would make a
runtime Tensor/private backend value look like a valid compiler key.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any, cast

from tllm.ports.address import LocatorFieldDesc, PortAddressDesc
from tllm.ports.port_id import PortId
from tllm.workgraph.ir import (
    AccessDesc,
    AnchorDesc,
    BackendSnapshot,
    CompiledWorkGraph,
    DataCentricIR,
    ExecutionNodeDesc,
    ExecutorKey,
    LeaseDesc,
    MoveDesc,
    NodeKey,
    ResidenceDesc,
    RuntimePlanBinding,
    SemanticGraph,
    SemanticNodeDesc,
    TimingDesc,
    ValueDesc,
)
from tllm.workgraph.frontend import (
    ArtifactKey,
    CapturedFlow,
    CapturedValueDesc,
    DynamicDim,
    ExternContract,
    ExternInputDesc,
    FlowCaptureDiagnostic,
    LeafArtifactDesc,
    LeafOperatorDesc,
    OperatorAttributeDesc,
    PortBindingDesc,
    ShapeGuard,
    ShapeGuardExpr,
    SourceSpan,
    TensorDTypeDesc,
    TensorDeviceDesc,
    TypeDesc,
    UseDefDesc,
)
from tllm.workgraph.hierarchy import (
    CaptureAttachment,
    CaptureDataCentricIR,
    CaptureHierarchy,
    CaptureLeaseDesc,
    LeafRegionDesc,
    RegionKey,
    RootFlowDesc,
    TaskRegionDesc,
)
from tllm.workgraph.physical import (
    BufferBucketDesc,
    BufferBucketKey,
    FenceDependencyDesc,
    FenceKey,
    PhysicalFlowPlan,
    PhysicalNodePlan,
)
from tllm.workgraph.placement import (
    CopyEngineCapacityDesc,
    DeviceTopology,
    DurationProfile,
    ImplementationDesc,
    ImplementationSelectionDesc,
    Move,
    MoveKey,
    PlacedNodeDesc,
    Placement,
    PlacementLeaseDesc,
    PlacementPlan,
    Residency,
    ScheduledMove,
    SemanticOracleKey,
    SemanticOracleVerificationDesc,
    TransferProfile,
    TransferProfileDesc,
    ValueFlowDesc,
    ValueKey,
)
from tllm.workgraph.phase4 import (
    CollectiveDesc,
    CollectiveKey,
    CollectiveRequestDesc,
    CommunicatorDesc,
    CommunicatorKey,
    DistributedFlowPlan,
    DistributedTopology,
    LocalCollectiveSchedule,
    PlacementAffinity,
    ShardSpec,
    StateKey,
    StateRequirementDesc,
    StateResidenceDesc,
    TopologyRankDesc,
)
from tllm.workgraph.phase5 import (
    BudgetUseDesc,
    CandidateActionDesc,
    CandidateActionKey,
    CandidateMoveSetDesc,
    EWMAProfileKey,
    FixedRatio,
    IntegerEWMAProfile,
    InterferenceBudgetDesc,
    InterferenceBudgetKey,
    MoveSetKey,
    Phase5ControlPlan,
    ShapeBucketDesc,
    ShapeBucketKey,
    WorkItemDesc,
    WorkItemKey,
    WorkloadBucketDesc,
    WorkloadBucketKey,
)


def _unsupported(value: Any) -> TypeError:
    value_type = cast(type[Any], type(value))
    return TypeError(
        "structural metadata must contain only the declared WorkGraph "
        "descriptors, primitives, Enum, PortId, PortAddressDesc, or tuples; "
        f"got {value_type.__module__}.{value_type.__qualname__}"
    )


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, PortId):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "version": value.version,
        }
    if isinstance(value, LocatorFieldDesc):
        return {
            "path": _canonical(value.path),
            "value_type": value.value_type,
            "value": value.value,
        }
    if isinstance(value, PortAddressDesc):
        return {
            "kind": value.kind,
            "locator_type": value.locator_type,
            "locator_fields": _canonical(value.locator_fields),
        }
    if isinstance(value, NodeKey):
        return {
            "namespace": value.namespace,
            "owner": value.owner,
            "ordinal": value.ordinal,
        }
    if isinstance(value, ExecutorKey):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "version": value.version,
        }
    if isinstance(value, ArtifactKey):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "version": value.version,
            "digest": value.digest,
        }
    if isinstance(value, FenceKey):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "ordinal": value.ordinal,
        }
    if isinstance(value, BufferBucketKey):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "ordinal": value.ordinal,
        }
    if isinstance(value, BufferBucketDesc):
        return {
            "key": _canonical(value.key),
            "shape": _canonical(value.shape),
            "dtype": _canonical(value.dtype),
            "device": _canonical(value.device),
            "slots": value.slots,
            "ownership": _canonical(value.ownership),
        }
    if isinstance(value, FenceDependencyDesc):
        return {
            "fence": _canonical(value.fence),
            "producer": _canonical(value.producer),
            "consumer": _canonical(value.consumer),
            "kind": _canonical(value.kind),
        }
    if isinstance(value, PhysicalNodePlan):
        return {
            "node_key": _canonical(value.node_key),
            "executor": _canonical(value.executor),
            "stream": _canonical(value.stream),
            "fences": _canonical(value.fences),
            "artifact_key": _canonical(value.artifact_key),
            "buffer_bucket": _canonical(value.buffer_bucket),
            "capture_policy": _canonical(value.capture_policy),
            "fallback_policy": _canonical(value.fallback_policy),
        }
    if isinstance(value, PhysicalFlowPlan):
        return {
            "plan_epoch": value.plan_epoch,
            "nodes": _canonical(value.nodes),
            "buffer_buckets": _canonical(value.buffer_buckets),
            "schema_version": value.schema_version,
        }
    if isinstance(value, Placement):
        return {"kind": _canonical(value.kind), "index": value.index}
    if isinstance(value, DeviceTopology):
        return {
            "peer_access": _canonical(tuple(sorted(value.peer_access))),
            "cuda_devices": _canonical(value.cuda_devices),
        }
    if isinstance(value, ValueKey):
        return {"name": value.name, "version": value.version}
    if isinstance(value, Residency):
        return {
            "value": _canonical(value.value),
            "placement": _canonical(value.placement),
            "memory": _canonical(value.memory),
        }
    if isinstance(value, Move):
        return {
            "value": _canonical(value.value),
            "source": _canonical(value.source),
            "destination": _canonical(value.destination),
            "kind": _canonical(value.kind),
            "byte_count": value.byte_count,
            "asynchronous": value.asynchronous,
        }
    if isinstance(value, MoveKey):
        return {"value": _canonical(value.value), "ordinal": value.ordinal}
    if isinstance(value, DurationProfile):
        return {"microseconds": value.microseconds, "samples": value.samples}
    if isinstance(value, TransferProfile):
        return {
            "h2d": _canonical(value.h2d),
            "d2h": _canonical(value.d2h),
            "p2p": _canonical(value.p2p),
            "fallback": _canonical(value.fallback),
        }
    if isinstance(value, TransferProfileDesc):
        return {
            "kind": _canonical(value.kind),
            "source": _canonical(value.source),
            "destination": _canonical(value.destination),
            "bytes_per_microsecond": value.bytes_per_microsecond,
            "fixed_latency_microseconds": value.fixed_latency_microseconds,
            "samples": value.samples,
        }
    if isinstance(value, CopyEngineCapacityDesc):
        return {"engine": _canonical(value.engine), "capacity": value.capacity}
    if isinstance(value, ValueFlowDesc):
        return {
            "value": _canonical(value.value),
            "producer": _canonical(value.producer),
            "consumer": _canonical(value.consumer),
            "byte_count": value.byte_count,
        }
    if isinstance(value, ImplementationDesc):
        return {
            "key": _canonical(value.key),
            "logical_executor": _canonical(value.logical_executor),
            "placement": _canonical(value.placement),
            "duration": _canonical(value.duration),
            "oracle_key": _canonical(value.oracle_key),
        }
    if isinstance(value, ImplementationSelectionDesc):
        return {
            "node_key": _canonical(value.node_key),
            "logical_executor": _canonical(value.logical_executor),
            "key": _canonical(value.key),
            "placement": _canonical(value.placement),
            "duration": _canonical(value.duration),
            "oracle_key": _canonical(value.oracle_key),
        }
    if isinstance(value, SemanticOracleKey):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "version": value.version,
        }
    if isinstance(value, SemanticOracleVerificationDesc):
        return {
            "logical_executor": _canonical(value.logical_executor),
            "oracle_key": _canonical(value.oracle_key),
            "implementations": _canonical(value.implementations),
            "case_count": value.case_count,
        }
    if isinstance(value, PlacedNodeDesc):
        return {
            "node_key": _canonical(value.node_key),
            "implementation": _canonical(value.implementation),
            "placement": _canonical(value.placement),
            "start_at": value.start_at,
            "end_at": value.end_at,
            "depends_on": _canonical(value.depends_on),
        }
    if isinstance(value, ScheduledMove):
        return {
            "key": _canonical(value.key),
            "move": _canonical(value.move),
            "engine": _canonical(value.engine),
            "start_at": value.start_at,
            "end_at": value.end_at,
            "producer": _canonical(value.producer),
            "consumer": _canonical(value.consumer),
            "depends_on": _canonical(value.depends_on),
            "used_fallback": value.used_fallback,
        }
    if isinstance(value, PlacementLeaseDesc):
        return {
            "residency": _canonical(value.residency),
            "live_start": value.live_start,
            "live_end": value.live_end,
        }
    if isinstance(value, PlacementPlan):
        return {
            "plan_epoch": value.plan_epoch,
            "selections": _canonical(value.selections),
            "schedule": _canonical(value.schedule),
            "topology": _canonical(value.topology),
            "value_flows": _canonical(value.value_flows),
            "residencies": _canonical(value.residencies),
            "moves": _canonical(value.moves),
            "leases": _canonical(value.leases),
            "copy_engines": _canonical(value.copy_engines),
            "schema_version": value.schema_version,
        }
    if isinstance(value, StateKey):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "version": value.version,
        }
    if isinstance(value, ShardSpec):
        return {
            "kind": _canonical(value.kind),
            "partitions": value.partitions,
            "axis": value.axis,
        }
    if isinstance(value, PlacementAffinity):
        return {
            "ranks": _canonical(value.ranks),
            "node_ids": _canonical(value.node_ids),
            "role": _canonical(value.role),
        }
    if isinstance(value, StateRequirementDesc):
        return {
            "key": _canonical(value.key),
            "scope": _canonical(value.scope),
            "shard": _canonical(value.shard),
            "affinity": _canonical(value.affinity),
        }
    if isinstance(value, TopologyRankDesc):
        return {
            "rank": value.rank,
            "node_id": value.node_id,
            "local_rank": value.local_rank,
            "placement": _canonical(value.placement),
            "role": _canonical(value.role),
            "tp_rank": value.tp_rank,
            "pp_rank": value.pp_rank,
            "dp_rank": value.dp_rank,
        }
    if isinstance(value, DistributedTopology):
        return {
            "ranks": _canonical(value.ranks),
            "tp_size": value.tp_size,
            "pp_size": value.pp_size,
            "dp_size": value.dp_size,
        }
    if isinstance(value, StateResidenceDesc):
        return {
            "key": _canonical(value.key),
            "rank": value.rank,
            "placement": _canonical(value.placement),
            "shard_index": value.shard_index,
            "replica_index": value.replica_index,
        }
    if isinstance(value, CommunicatorKey):
        return {"kind": _canonical(value.kind), "group_index": value.group_index}
    if isinstance(value, CommunicatorDesc):
        return {
            "key": _canonical(value.key),
            "ranks": _canonical(value.ranks),
            "backend": _canonical(value.backend),
        }
    if isinstance(value, CollectiveKey):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "ordinal": value.ordinal,
        }
    if isinstance(value, CollectiveRequestDesc):
        return {
            "key": _canonical(value.key),
            "kind": _canonical(value.kind),
            "group": _canonical(value.group),
            "group_index": value.group_index,
            "backend": _canonical(value.backend),
            "window": _canonical(value.window),
            "value": _canonical(value.value),
            "depends_on": _canonical(value.depends_on),
            "root_rank": value.root_rank,
        }
    if isinstance(value, CollectiveDesc):
        return {
            "key": _canonical(value.key),
            "kind": _canonical(value.kind),
            "communicator": _canonical(value.communicator),
            "sequence": value.sequence,
            "window": _canonical(value.window),
            "value": _canonical(value.value),
            "depends_on": _canonical(value.depends_on),
            "root_rank": value.root_rank,
        }
    if isinstance(value, LocalCollectiveSchedule):
        return {
            "rank": value.rank,
            "collectives": _canonical(value.collectives),
        }
    if isinstance(value, DistributedFlowPlan):
        return {
            "plan_epoch": value.plan_epoch,
            "topology": _canonical(value.topology),
            "state_requirements": _canonical(value.state_requirements),
            "state_residences": _canonical(value.state_residences),
            "collective_requests": _canonical(value.collective_requests),
            "communicators": _canonical(value.communicators),
            "collectives": _canonical(value.collectives),
            "local_schedules": _canonical(value.local_schedules),
            "schema_version": value.schema_version,
        }
    if isinstance(value, (EWMAProfileKey, WorkloadBucketKey, ShapeBucketKey, InterferenceBudgetKey)):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "version": value.version,
        }
    if isinstance(value, (MoveSetKey, CandidateActionKey, WorkItemKey)):
        return {
            "namespace": value.namespace,
            "name": value.name,
            "ordinal": value.ordinal,
        }
    if isinstance(value, FixedRatio):
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    if isinstance(value, IntegerEWMAProfile):
        return {
            "key": _canonical(value.key),
            "estimate": value.estimate,
            "alpha": _canonical(value.alpha),
            "samples": value.samples,
            "minimum": value.minimum,
            "maximum": value.maximum,
        }
    if isinstance(value, ShapeBucketDesc):
        return {
            "key": _canonical(value.key),
            "shape": _canonical(value.shape),
            "lower_bounds": _canonical(value.lower_bounds),
            "upper_bounds": _canonical(value.upper_bounds),
        }
    if isinstance(value, WorkloadBucketDesc):
        return {
            "key": _canonical(value.key),
            "shape_buckets": _canonical(value.shape_buckets),
            "required_by_default": value.required_by_default,
        }
    if isinstance(value, InterferenceBudgetDesc):
        return {
            "key": _canonical(value.key),
            "capacity": value.capacity,
            "profile": _canonical(value.profile),
        }
    if isinstance(value, BudgetUseDesc):
        return {"key": _canonical(value.key), "amount": value.amount}
    if isinstance(value, CandidateMoveSetDesc):
        return {"key": _canonical(value.key), "moves": _canonical(value.moves)}
    if isinstance(value, CandidateActionDesc):
        return {
            "key": _canonical(value.key),
            "kind": _canonical(value.kind),
            "semantic_key": value.semantic_key,
            "workload_bucket": _canonical(value.workload_bucket),
            "shape_bucket": _canonical(value.shape_bucket),
            "move_set": _canonical(value.move_set),
            "fallback_key": _canonical(value.fallback_key),
            "deadline": value.deadline,
            "ready_at": value.ready_at,
            "duration": value.duration,
            "buffer_credits": value.buffer_credits,
            "budget_uses": _canonical(value.budget_uses),
            "priority": value.priority,
            "utility": value.utility,
            "required": value.required,
            "unknown_bucket": value.unknown_bucket,
            "authoritative": value.authoritative,
        }
    if isinstance(value, WorkItemDesc):
        return {
            "key": _canonical(value.key),
            "shape": _canonical(value.shape),
            "workload_bucket": _canonical(value.workload_bucket),
            "semantic_key": value.semantic_key,
            "required": value.required,
            "allow_shedding": value.allow_shedding,
            "deadline": value.deadline,
        }
    if isinstance(value, Phase5ControlPlan):
        return {
            "plan_epoch": value.plan_epoch,
            "placement_plan": _canonical(value.placement_plan),
            "placement_signature": value.placement_signature,
            "scheduled_moves": _canonical(value.scheduled_moves),
            "move_sets": _canonical(value.move_sets),
            "workload_buckets": _canonical(value.workload_buckets),
            "shape_buckets": _canonical(value.shape_buckets),
            "budgets": _canonical(value.budgets),
            "actions": _canonical(value.actions),
            "unknown_bucket_fallback": _canonical(value.unknown_bucket_fallback),
            "buffer_capacity": value.buffer_capacity,
            "mode": _canonical(value.mode),
            "profiles": _canonical(value.profiles),
            "densification_limit": value.densification_limit,
            "schema_version": value.schema_version,
        }
    if isinstance(value, SourceSpan):
        return {
            "file": value.file,
            "start_line": value.start_line,
            "start_column": value.start_column,
            "end_line": value.end_line,
            "end_column": value.end_column,
        }
    if isinstance(value, ShapeGuardExpr):
        return {
            "value_path": _canonical(value.value_path),
            "dimension": value.dimension,
            "kind": _canonical(value.kind),
            "lower_bound": value.lower_bound,
            "upper_bound": value.upper_bound,
        }
    if isinstance(value, ShapeGuard):
        return {"expressions": _canonical(value.expressions)}
    if isinstance(value, DynamicDim):
        return {
            "value_path": _canonical(value.value_path),
            "dimension": value.dimension,
            "lower_bound": value.lower_bound,
            "upper_bound": value.upper_bound,
        }
    if isinstance(value, TensorDeviceDesc):
        return {"kind": _canonical(value.kind), "index": value.index}
    if isinstance(value, TensorDTypeDesc):
        return {"namespace": value.namespace, "name": value.name}
    if isinstance(value, UseDefDesc):
        return {
            "producer": value.producer,
            "inputs": _canonical(value.inputs),
            "users": _canonical(value.users),
        }
    if isinstance(value, CapturedValueDesc):
        return {
            "value_id": value.value_id,
            "version": value.version,
            "shape": _canonical(value.shape),
            "dtype": _canonical(value.dtype),
            "device": _canonical(value.device),
            "layout": _canonical(value.layout),
            "stride": _canonical(value.stride),
            "use_def": _canonical(value.use_def),
            "alias_set": value.alias_set,
            "effect": _canonical(value.effect),
        }
    if isinstance(value, LeafOperatorDesc):
        return {
            "ordinal": value.ordinal,
            "namespace": value.namespace,
            "name": value.name,
            "overload": value.overload,
            "inputs": _canonical(value.inputs),
            "outputs": _canonical(value.outputs),
            "module_path": _canonical(value.module_path),
            "operator_kind": _canonical(value.operator_kind),
            "effects": _canonical(value.effects),
            "capabilities": _canonical(value.capabilities),
            "attributes": _canonical(value.attributes),
        }
    if isinstance(value, OperatorAttributeDesc):
        return {
            "path": _canonical(value.path),
            "kind": _canonical(value.kind),
            "value": value.value,
        }
    if isinstance(value, LeafArtifactDesc):
        return {
            "artifact_key": _canonical(value.artifact_key),
            "operators": _canonical(value.operators),
            "input_values": _canonical(value.input_values),
            "output_values": _canonical(value.output_values),
            "terminal_ordinal": value.terminal_ordinal,
        }
    if isinstance(value, PortBindingDesc):
        return {
            "port": _canonical(value.port),
            "direction": _canonical(value.direction),
            "value_path": _canonical(value.value_path),
        }
    if isinstance(value, FlowCaptureDiagnostic):
        return {
            "code": _canonical(value.code),
            "message": value.message,
            "source_span": _canonical(value.source_span),
            "detail": value.detail,
        }
    if isinstance(value, TypeDesc):
        return {"module": value.module, "qualname": value.qualname}
    if isinstance(value, ExternInputDesc):
        return {
            "position": value.position,
            "name": value.name,
            "value_type": _canonical(value.value_type),
        }
    if isinstance(value, ExternContract):
        return {
            "inputs": _canonical(value.inputs),
            "output": _canonical(value.output),
            "effect": _canonical(value.effect),
            "state": _canonical(value.state),
            "device": _canonical(value.device),
            "capability": _canonical(value.capability),
            "bindings": _canonical(value.bindings),
        }
    if isinstance(value, CapturedFlow):
        return {
            "artifact_key": _canonical(value.artifact_key),
            "source_span": _canonical(value.source_span),
            "shape_guard": _canonical(value.shape_guard),
            "inputs": _canonical(value.inputs),
            "outputs": _canonical(value.outputs),
            "values": _canonical(value.values),
            "leaf_artifact": _canonical(value.leaf_artifact),
            "effects": _canonical(value.effects),
            "diagnostics": _canonical(value.diagnostics),
            "status": _canonical(value.status),
            "node_kind": _canonical(value.node_kind),
            "mobility": _canonical(value.mobility),
            "graph_safety": _canonical(value.graph_safety),
        }
    if isinstance(value, RegionKey):
        return {
            "artifact_key": _canonical(value.artifact_key),
            "kind": _canonical(value.kind),
            "ordinal": value.ordinal,
        }
    if isinstance(value, LeafRegionDesc):
        return {
            "key": _canonical(value.key),
            "artifact_key": _canonical(value.artifact_key),
            "kind": _canonical(value.kind),
            "operator_ordinals": _canonical(value.operator_ordinals),
            "leaf_artifact": _canonical(value.leaf_artifact),
        }
    if isinstance(value, TaskRegionDesc):
        return {
            "key": _canonical(value.key),
            "node_key": _canonical(value.node_key),
            "executor": _canonical(value.executor),
            "boundary": _canonical(value.boundary),
            "module_path": _canonical(value.module_path),
            "operator_ordinals": _canonical(value.operator_ordinals),
            "input_values": _canonical(value.input_values),
            "output_values": _canonical(value.output_values),
            "effects": _canonical(value.effects),
            "capabilities": _canonical(value.capabilities),
            "dependencies": _canonical(value.dependencies),
            "children": _canonical(value.children),
            "kind": _canonical(value.kind),
        }
    if isinstance(value, RootFlowDesc):
        return {
            "key": _canonical(value.key),
            "artifact_key": _canonical(value.artifact_key),
            "source_span": _canonical(value.source_span),
            "status": _canonical(value.status),
            "shape_guard": _canonical(value.shape_guard),
            "inputs": _canonical(value.inputs),
            "outputs": _canonical(value.outputs),
            "values": _canonical(value.values),
            "effects": _canonical(value.effects),
            "diagnostics": _canonical(value.diagnostics),
            "children": _canonical(value.children),
            "kind": _canonical(value.kind),
        }
    if isinstance(value, CaptureAttachment):
        return {
            "node_key": _canonical(value.node_key),
            "artifact_key": _canonical(value.artifact_key),
            "bindings": _canonical(value.bindings),
            "anchor_node_key": _canonical(value.anchor_node_key),
        }
    if isinstance(value, CaptureLeaseDesc):
        return {
            "value_id": value.value_id,
            "live_start": _canonical(value.live_start),
            "live_end": _canonical(value.live_end),
            "reader_count": value.reader_count,
            "materialized": value.materialized,
        }
    if isinstance(value, CaptureDataCentricIR):
        return {
            "leases": _canonical(value.leases),
            "moves": _canonical(value.moves),
        }
    if isinstance(value, CaptureHierarchy):
        return {
            "schema_version": value.schema_version,
            "roots": _canonical(value.roots),
            "attachments": _canonical(value.attachments),
            "data_centric": _canonical(value.data_centric),
        }
    if isinstance(value, BackendSnapshot):
        return {
            "backend_id": value.backend_id,
            "runner_family": value.runner_family,
            "graph_mode": value.graph_mode,
            "backend_version": value.backend_version,
            "capabilities": _canonical(value.capabilities),
            "device": value.device,
            "rank": value.rank,
            "tp_size": value.tp_size,
            "pp_size": value.pp_size,
            "dp_size": value.dp_size,
            "topology": _canonical(value.topology),
        }
    if isinstance(value, AnchorDesc):
        return {
            "backend": value.backend,
            "runner": value.runner,
            "graph_mode": value.graph_mode,
            "hook": value.hook,
            "externally_owned": value.externally_owned,
            "mobility": value.mobility,
            "service_class": value.service_class,
        }
    if isinstance(value, TimingDesc):
        return {
            "window": value.window,
            "mobility": value.mobility,
            "deadline": value.deadline,
            "max_defer_steps": value.max_defer_steps,
            "observes_version": value.observes_version,
        }
    if isinstance(value, ValueDesc):
        return {
            "value_id": value.value_id,
            "version": value.version,
            "port": _canonical(value.port),
            "shape": _canonical(value.shape),
            "dtype": value.dtype,
            "layout": value.layout,
            "alias_set": value.alias_set,
        }
    if isinstance(value, AccessDesc):
        return {
            "node_key": _canonical(value.node_key),
            "value": _canonical(value.value),
            "mode": _canonical(value.mode),
            "port": _canonical(value.port),
            "subset": _canonical(value.subset),
            "consistency": value.consistency,
        }
    if isinstance(value, ResidenceDesc):
        return {
            "value": _canonical(value.value),
            "device": value.device,
            "rank": value.rank,
            "memory_kind": value.memory_kind,
            "valid_version": value.valid_version,
        }
    if isinstance(value, MoveDesc):
        return {
            "value": _canonical(value.value),
            "source": _canonical(value.source),
            "destination": _canonical(value.destination),
            "subset": _canonical(value.subset),
            "layout": value.layout,
            "engine": value.engine,
            "asynchronous": value.asynchronous,
        }
    if isinstance(value, LeaseDesc):
        return {
            "value": _canonical(value.value),
            "storage": value.storage,
            "live_start": value.live_start,
            "live_end": value.live_end,
            "ready_fence": value.ready_fence,
            "retire_fence": value.retire_fence,
            "reader_count": value.reader_count,
            "stable_address": value.stable_address,
            "materialized": value.materialized,
        }
    if isinstance(value, SemanticNodeDesc):
        return {
            "node_key": _canonical(value.node_key),
            "kind": _canonical(value.kind),
            "executor": _canonical(value.executor),
            "reads": _canonical(value.reads),
            "writes": _canonical(value.writes),
            "effects": _canonical(value.effects),
            "timing": _canonical(value.timing),
            "anchor": _canonical(value.anchor),
            "graph_safety": _canonical(value.graph_safety),
            "source": value.source,
            "legacy_boundary": _canonical(value.legacy_boundary),
            "shape_guard": _canonical(value.shape_guard),
        }
    if isinstance(value, SemanticGraph):
        return {
            "schema_version": value.schema_version,
            "nodes": _canonical(value.nodes),
            "edges": _canonical(value.edges),
            "values": _canonical(value.values),
            "backend": _canonical(value.backend),
            "graph_mode": value.graph_mode,
        }
    if isinstance(value, ExecutionNodeDesc):
        return {
            "node_key": _canonical(value.node_key),
            "semantic_key": _canonical(value.semantic_key),
            "kind": _canonical(value.kind),
            "executor": _canonical(value.executor),
            "accesses": _canonical(value.accesses),
            "residence_keys": _canonical(value.residence_keys),
            "move_keys": _canonical(value.move_keys),
            "lease_keys": _canonical(value.lease_keys),
            "device": value.device,
            "rank": value.rank,
            "anchor": _canonical(value.anchor),
            "graph_safety": _canonical(value.graph_safety),
            "fusion_group": value.fusion_group,
        }
    if isinstance(value, DataCentricIR):
        return {
            "schema_version": value.schema_version,
            "nodes": _canonical(value.nodes),
            "values": _canonical(value.values),
            "accesses": _canonical(value.accesses),
            "residences": _canonical(value.residences),
            "moves": _canonical(value.moves),
            "leases": _canonical(value.leases),
            "edges": _canonical(value.edges),
            "backend": _canonical(value.backend),
            "graph_mode": value.graph_mode,
        }
    if isinstance(value, RuntimePlanBinding):
        return {
            "dispatch_flow_ids": _canonical(value.dispatch_flow_ids),
            "dispatch_hooks": _canonical(value.dispatch_hooks),
            "dispatch_fusion_groups": _canonical(value.dispatch_fusion_groups),
            "execution_producer_ids": _canonical(value.execution_producer_ids),
            "producer_hooks": _canonical(value.producer_hooks),
            "execution_port_sources": _canonical(value.execution_port_sources),
            "capture_resident_producer_ids": _canonical(value.capture_resident_producer_ids),
        }
    if isinstance(value, CompiledWorkGraph):
        # Plan epoch and a cached digest are revision metadata, not graph
        # structure.  The two IR layers are the complete signature input.
        return _canonical((value.semantic, value.data_centric))
    if isinstance(value, tuple):
        tuple_value = cast(tuple[Any, ...], value)
        return [_canonical(item) for item in tuple_value]
    raise _unsupported(value)


def canonical_value(value: Any) -> Any:
    """Return the JSON-compatible canonical structural representation."""

    return _canonical(value)


def canonical_json(value: Any) -> str:
    """Serialize metadata with stable key ordering and no runtime payload."""

    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def structural_signature(value: Any) -> str:
    """Return the full SHA-256 digest of canonical structural metadata."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def signature_payload(value: Any) -> Any:
    """Return the canonical payload used by the signature oracle."""

    return canonical_value(value)


sha256_signature = structural_signature
