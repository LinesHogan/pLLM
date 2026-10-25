"""Shared immutable graph/plan data; Rust owns semantic validation at compilation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, NewType

from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.region_imports import RegionTensorImport
from pllm.contracts.region_program import Program
from pllm.contracts.region_selection import RegionSelectionPolicy
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.visibility import VisibilitySource

WorkloadId = NewType("WorkloadId", str)
TaskSymbol = NewType("TaskSymbol", str)
ImplementationId = NewType("ImplementationId", str)
AnchorSymbol = NewType("AnchorSymbol", str)
TaskId = NewType("TaskId", int)
AnchorId = NewType("AnchorId", int)
StateResource = NewType("StateResource", str)
HostServiceId = NewType("HostServiceId", str)
HostServicePort = NewType("HostServicePort", str)
HostPayloadSchemaId = NewType("HostPayloadSchemaId", str)


class ExecutionKind(StrEnum):
    HOST_SERVICE = "host_service"
    NATIVE_TASK = "native_task"


class StateScope(StrEnum):
    REQUEST = "request"
    INSTANCE = "instance"


class SemanticOutputKind(StrEnum):
    HOST = "host"
    TEMPORARY = "temporary"
    SAMPLER = "sampler"


class StateEffectKind(StrEnum):
    READ = "read"
    TRANSITION = "transition"
    TRANSITION_MAYBE = "transition_maybe"
    UPDATE = "update"
    UPDATE_MAYBE = "update_maybe"
    PUBLISH = "publish"


@dataclass(frozen=True, slots=True)
class HiddenInputSource:
    layer: int
    site: HiddenSite
    phase: HiddenPhase


@dataclass(frozen=True, slots=True)
class StepIdInputSource:
    pass


@dataclass(frozen=True, slots=True)
class PackedRowsInputSource:
    pass


@dataclass(frozen=True, slots=True)
class PackedSamplerRowsInputSource:
    pass


@dataclass(frozen=True, slots=True)
class SamplerFeedbackInputSource:
    pass


@dataclass(frozen=True, slots=True)
class HostServiceInputSource:
    service: HostServiceId
    port: HostServicePort
    schema: HostPayloadSchemaId


@dataclass(frozen=True, slots=True)
class StateSelectionInputSource:
    resource: StateResource
    phase: HiddenPhase


@dataclass(frozen=True, slots=True)
class TaskResultInputSource:
    task: TaskSymbol
    output: str


SemanticInputSource = (
    HiddenInputSource
    | StepIdInputSource
    | PackedRowsInputSource
    | PackedSamplerRowsInputSource
    | SamplerFeedbackInputSource
    | HostServiceInputSource
    | StateSelectionInputSource
    | TaskResultInputSource
)


@dataclass(frozen=True, slots=True)
class SemanticInput:
    name: str
    source: SemanticInputSource


@dataclass(frozen=True, slots=True)
class SemanticOutput:
    name: str
    kind: SemanticOutputKind = SemanticOutputKind.HOST
    schema: HostPayloadSchemaId = HostPayloadSchemaId("")


@dataclass(frozen=True, slots=True)
class SemanticState:
    resource: StateResource
    scope: StateScope
    key_fields: tuple[RequestKeyField, ...]


@dataclass(frozen=True, slots=True)
class SemanticEffect:
    resource: StateResource
    kind: StateEffectKind
    # None names the invocation's input version, not an implicit mutable latest.
    version: TaskSymbol | None = None


@dataclass(frozen=True, slots=True)
class TimingConstraint:
    after: AnchorSymbol
    before: AnchorSymbol


@dataclass(frozen=True, slots=True)
class TaskImplementationSpec:
    implementation_id: ImplementationId
    execution_kind: ExecutionKind
    required_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConditionalLayerReplay:
    """Seed from hidden+residual at the first replayed layer; carry each detached block output.

    `layers` is the signed ascending set of layers re-run, in order, for the
    selected token with the published state; each replayed layer overwrites its
    own current KV slot. Layers outside the set keep their normal-forward KV —
    a declared approximation, not a silent one."""

    layers: tuple[int, ...]
    max_iterations: int = 2
    relative_l2_ppm: int = 10
    predicate: Literal["weight_updated"] = "weight_updated"
    rows: Literal["last_scheduled_token"] = "last_scheduled_token"
    carry: Literal["block_output"] = "block_output"
    kv: Literal["overwrite_current_slot"] = "overwrite_current_slot"
    logits: Literal["preserve"] = "preserve"
    rng: Literal["preserve"] = "preserve"
    gradient: Literal["detach"] = "detach"
    cancellation: Literal["between_iterations_restore_kv"] = "between_iterations_restore_kv"


@dataclass(frozen=True, slots=True)
class TokenProgramEffect:
    """A token program's model domain and selected rows; mathematics lives in its source region."""

    layer_count: int
    rows: Literal["prefill", "decode", "all"] = "all"


LossReduction = Literal[
    "none", "sum", "mean", "state_mean", "weighted_state_mean", "weighted_state_mean_then_mean", "custom"
]


@dataclass(frozen=True, slots=True)
class MathContract:
    """Explicit differentiation and row domains of a native mathematical operation."""

    input_gradients: tuple[str, ...] = ()
    parameter_gradients: tuple[StateResource, ...] = ()
    detached_inputs: tuple[str, ...] = ()
    forward_rows: str = ""
    input_gradient_rows: str = ""
    parameter_gradient_rows: str = ""
    update_states: str = ""
    reduction: LossReduction = "none"


@dataclass(frozen=True, slots=True)
class SemanticTask:
    symbol: TaskSymbol
    timing: TimingConstraint
    inputs: tuple[SemanticInput, ...]
    outputs: tuple[SemanticOutput, ...]
    implementations: tuple[TaskImplementationSpec, ...]
    effects: tuple[SemanticEffect, ...] = ()
    replay: ConditionalLayerReplay | TokenProgramEffect | None = None
    math: MathContract | None = None
    visibility: VisibilitySource | None = None


@dataclass(frozen=True, slots=True)
class SemanticEdge:
    source: TaskSymbol
    target: TaskSymbol


@dataclass(frozen=True, slots=True)
class ConditionalInputGroup:
    """Atomic producer closure; absent input reads the invocation's pinned version.

    Existing edges/effects describe the complete-input branch and conservatively
    schedule both branches. Only the named readers may omit publication inputs.
    """

    resource: StateResource
    borrowed_resources: tuple[StateResource, ...]
    tasks: tuple[TaskSymbol, ...]
    publication: TaskSymbol
    readers: tuple[TaskSymbol, ...]
    allow_absent: bool
    held_version: Literal["input"] = "input"


@dataclass(frozen=True, slots=True)
class TaskSourceRegion:
    """Complete task-owned source, available before physical partitioning."""

    task: TaskSymbol
    name: str
    program: Program
    imports: tuple[RegionTensorImport, ...] = ()


@dataclass(frozen=True, slots=True)
class SemanticGraph:
    workload_id: WorkloadId
    tasks: tuple[SemanticTask, ...]
    execution_parameters_digest: str
    states: tuple[SemanticState, ...] = ()
    edges: tuple[SemanticEdge, ...] = ()
    schema_version: int = 1
    input_groups: tuple[ConditionalInputGroup, ...] = ()
    source_regions: tuple[TaskSourceRegion, ...] = ()


@dataclass(frozen=True, slots=True)
class TimelineAnchor:
    symbol: AnchorSymbol
    ordinal: int


@dataclass(frozen=True, slots=True)
class ValueResourceFact:
    task: TaskSymbol
    output: str
    bytes: int


@dataclass(frozen=True, slots=True)
class TaskResourceFact:
    task: TaskSymbol
    workspace_bytes: int
    resident_bytes: int = 0
    # Fixed cold buffers; native bindings define which input axes may be padded.
    input_shapes: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class SchedulerTimeline:
    anchors: tuple[TimelineAnchor, ...]
    capabilities: tuple[str, ...] = ()
    schema_version: int = 1
    value_resources: tuple[ValueResourceFact, ...] = ()
    task_resources: tuple[TaskResourceFact, ...] = ()
    memory_budget_bytes: int | None = None
    reserved_bytes: tuple[int, ...] = ()
    region_policy: RegionSelectionPolicy = RegionSelectionPolicy()
    visibility: VisibilitySource | None = None


@dataclass(frozen=True, slots=True)
class PhysicalTask:
    task_id: TaskId
    implementation_id: ImplementationId
    anchor_id: AnchorId
    release_values: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskRegionSchedule:
    """A source-derived instance order signed with the enclosing physical plan."""

    task: TaskSymbol
    name: str
    schedule_map: str


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    graph_digest: str
    timeline_digest: str
    tasks: tuple[PhysicalTask, ...]
    fallback_count: int
    signature: str
    region_ends: tuple[int, ...]
    materialized_values: tuple[int, ...]
    materialization_bytes: int
    peak_memory_bytes: int
    fence_count: int
    state_plan_digest: str | None
    source_schedules: tuple[TaskRegionSchedule, ...] = ()
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class PhysicalPlan:
    plan_epoch: int
    schedule_signature: str
    tasks: tuple[PhysicalTask, ...]
    fallback_count: int
    region_ends: tuple[int, ...]
    materialized_values: tuple[int, ...]
    source_schedules: tuple[TaskRegionSchedule, ...] = ()
    schema_version: int = 1


__all__ = [
    "ConditionalLayerReplay",
    "TaskResourceFact",
    "ValueResourceFact",
    "AnchorId",
    "AnchorSymbol",
    "ExecutionKind",
    "HiddenInputSource",
    "HostServiceId",
    "HostServiceInputSource",
    "HostServicePort",
    "HostPayloadSchemaId",
    "ImplementationId",
    "PhysicalPlan",
    "PhysicalTask",
    "PackedRowsInputSource",
    "PackedSamplerRowsInputSource",
    "SamplerFeedbackInputSource",
    "ScheduleDecision",
    "SchedulerTimeline",
    "SemanticEdge",
    "SemanticEffect",
    "SemanticGraph",
    "ConditionalInputGroup",
    "SemanticInput",
    "SemanticOutput",
    "SemanticOutputKind",
    "SemanticState",
    "SemanticTask",
    "StepIdInputSource",
    "StateEffectKind",
    "StateResource",
    "StateScope",
    "StateSelectionInputSource",
    "TaskId",
    "TaskImplementationSpec",
    "TaskSymbol",
    "TaskResultInputSource",
    "TimelineAnchor",
    "TimingConstraint",
    "WorkloadId",
]
