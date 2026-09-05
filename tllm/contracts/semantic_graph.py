"""Shared immutable graph/plan data; Rust owns semantic validation at compilation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from tllm.contracts.request_identity import RequestKeyField


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
class SemanticTask:
    symbol: TaskSymbol
    timing: TimingConstraint
    inputs: tuple[SemanticInput, ...]
    outputs: tuple[SemanticOutput, ...]
    implementations: tuple[TaskImplementationSpec, ...]
    effects: tuple[SemanticEffect, ...] = ()



@dataclass(frozen=True, slots=True)
class SemanticEdge:
    source: TaskSymbol
    target: TaskSymbol



@dataclass(frozen=True, slots=True)
class SemanticGraph:
    workload_id: WorkloadId
    tasks: tuple[SemanticTask, ...]
    execution_parameters_digest: str
    states: tuple[SemanticState, ...] = ()
    edges: tuple[SemanticEdge, ...] = ()
    schema_version: int = 1



@dataclass(frozen=True, slots=True)
class TimelineAnchor:
    symbol: AnchorSymbol
    ordinal: int



@dataclass(frozen=True, slots=True)
class SchedulerTimeline:
    anchors: tuple[TimelineAnchor, ...]
    capabilities: tuple[str, ...] = ()
    schema_version: int = 1



@dataclass(frozen=True, slots=True)
class ScheduledTask:
    task_id: TaskId
    task_symbol: TaskSymbol
    implementation_id: ImplementationId
    anchor_id: AnchorId
    anchor_symbol: AnchorSymbol


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    graph_digest: str
    timeline_digest: str
    tasks: tuple[ScheduledTask, ...]
    fallback_count: int
    signature: str
    schema_version: int = 1



@dataclass(frozen=True, slots=True)
class PhysicalTask:
    task_id: TaskId
    implementation_id: ImplementationId
    anchor_id: AnchorId


@dataclass(frozen=True, slots=True)
class PhysicalPlan:
    plan_epoch: int
    schedule_signature: str
    tasks: tuple[PhysicalTask, ...]
    fallback_count: int
    schema_version: int = 1



__all__ = [
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
    "ScheduleDecision",
    "ScheduledTask",
    "SchedulerTimeline",
    "SemanticEdge",
    "SemanticEffect",
    "SemanticGraph",
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
