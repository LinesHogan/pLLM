"""Clean-room semantic graph, schedule, and physical-plan contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from tllm.contracts.request_identity import (
    RequestKeyField,
    canonical_request_key_fields,
)


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


@dataclass(frozen=True, slots=True)
class HiddenInputSource:
    layer: int
    site: HiddenSite
    phase: HiddenPhase

    def __post_init__(self) -> None:
        if (
            type(self.layer) is not int
            or type(self.site) is not HiddenSite
            or type(self.phase) is not HiddenPhase
            or self.site is HiddenSite.NONE
            or self.phase is HiddenPhase.NONE
        ):
            raise TypeError("hidden input source requires a closed locator")


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

    def __post_init__(self) -> None:
        if (
            type(self.service) is not str
            or not self.service
            or type(self.port) is not str
            or not self.port
            or type(self.schema) is not str
            or not self.schema
        ):
            raise ValueError("host-service input requires closed service and port IDs")


@dataclass(frozen=True, slots=True)
class StateSelectionInputSource:
    resource: StateResource
    phase: HiddenPhase

    def __post_init__(self) -> None:
        if (
            not self.resource
            or type(self.phase) is not HiddenPhase
            or self.phase is HiddenPhase.NONE
        ):
            raise ValueError("state selection requires a resource and phase")


@dataclass(frozen=True, slots=True)
class TaskResultInputSource:
    task: TaskSymbol
    output: str

    def __post_init__(self) -> None:
        if not self.task or not self.output:
            raise ValueError("task result source requires task and output")


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

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("semantic input requires a name")
        match self.source:
            case (
                HiddenInputSource()
                | StepIdInputSource()
                | PackedRowsInputSource()
                | PackedSamplerRowsInputSource()
                | HostServiceInputSource()
                | StateSelectionInputSource()
                | TaskResultInputSource()
            ):
                pass
            case _:
                raise TypeError("semantic input requires a closed source")


@dataclass(frozen=True, slots=True)
class SemanticOutput:
    name: str
    kind: SemanticOutputKind = SemanticOutputKind.HOST
    schema: HostPayloadSchemaId = HostPayloadSchemaId("")

    def __post_init__(self) -> None:
        if (
            not self.name
            or type(self.kind) is not SemanticOutputKind
            or type(self.schema) is not str
            or (
                self.kind is SemanticOutputKind.HOST
                and not self.schema
            )
            or (
                self.kind is not SemanticOutputKind.HOST
                and bool(self.schema)
            )
        ):
            raise ValueError("semantic output requires a name and closed kind")


@dataclass(frozen=True, slots=True)
class SemanticState:
    resource: StateResource
    scope: StateScope
    key_fields: tuple[RequestKeyField, ...]

    def __post_init__(self) -> None:
        if type(self.scope) is not StateScope:
            raise ValueError("semantic state requires a closed scope")
        match self.scope:
            case StateScope.REQUEST:
                canonical_request_key_fields(self.key_fields)
            case StateScope.INSTANCE:
                if type(self.key_fields) is not tuple or self.key_fields:
                    raise ValueError("instance state cannot declare request keys")
        if not self.resource:
            raise ValueError(
                "semantic state requires a resource and scope-compatible keys"
            )


@dataclass(frozen=True, slots=True)
class SemanticEffect:
    resource: StateResource
    kind: StateEffectKind

    def __post_init__(self) -> None:
        if not self.resource or type(self.kind) is not StateEffectKind:
            raise ValueError("semantic effect requires a resource and closed kind")


@dataclass(frozen=True, slots=True)
class TimingConstraint:
    after: AnchorSymbol
    before: AnchorSymbol

    def __post_init__(self) -> None:
        if not self.after or not self.before or self.after == self.before:
            raise ValueError("timing constraint requires distinct anchors")


@dataclass(frozen=True, slots=True)
class TaskImplementationSpec:
    implementation_id: ImplementationId
    execution_kind: ExecutionKind
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.implementation_id
            or type(self.execution_kind) is not ExecutionKind
        ):
            raise ValueError(
                "task implementation requires a stable ID and closed execution kind"
            )
        if (
            tuple(sorted(self.required_capabilities))
            != self.required_capabilities
            or len(set(self.required_capabilities))
            != len(self.required_capabilities)
            or any(not capability for capability in self.required_capabilities)
        ):
            raise ValueError(
                "task implementation capabilities must be sorted unique names"
            )


@dataclass(frozen=True, slots=True)
class SemanticTask:
    symbol: TaskSymbol
    timing: TimingConstraint
    inputs: tuple[SemanticInput, ...]
    outputs: tuple[SemanticOutput, ...]
    implementations: tuple[TaskImplementationSpec, ...]
    effects: tuple[SemanticEffect, ...] = ()

    def __post_init__(self) -> None:
        if not self.symbol or not self.implementations:
            raise ValueError("semantic task requires a symbol and implementations")
        input_names = tuple(item.name for item in self.inputs)
        output_names = tuple(item.name for item in self.outputs)
        implementation_ids = tuple(
            item.implementation_id for item in self.implementations
        )
        effect_resources = tuple(effect.resource for effect in self.effects)
        if (
            len(set(input_names)) != len(input_names)
            or len(set(output_names)) != len(output_names)
            or tuple(sorted(implementation_ids)) != implementation_ids
            or len(set(implementation_ids)) != len(implementation_ids)
            or len(set(effect_resources)) != len(effect_resources)
        ):
            raise ValueError("semantic task ports or implementations are not canonical")


@dataclass(frozen=True, slots=True)
class SemanticEdge:
    source: TaskSymbol
    target: TaskSymbol

    def __post_init__(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("semantic edge requires distinct task symbols")


@dataclass(frozen=True, slots=True)
class SemanticGraph:
    workload_id: WorkloadId
    tasks: tuple[SemanticTask, ...]
    execution_parameters_digest: str
    states: tuple[SemanticState, ...] = ()
    edges: tuple[SemanticEdge, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        symbols = tuple(task.symbol for task in self.tasks)
        resources = tuple(state.resource for state in self.states)
        edges = tuple((edge.source, edge.target) for edge in self.edges)
        if (
            not self.workload_id
            or not symbols
            or type(self.execution_parameters_digest) is not str
            or len(self.execution_parameters_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.execution_parameters_digest
            )
            or tuple(sorted(symbols)) != symbols
            or len(set(symbols)) != len(symbols)
            or tuple(sorted(resources)) != resources
            or len(set(resources)) != len(resources)
            or tuple(sorted(edges)) != edges
            or len(set(edges)) != len(edges)
            or self.schema_version != 1
        ):
            raise ValueError("semantic graph is not canonical")
        known = set(symbols)
        if any(edge.source not in known or edge.target not in known for edge in self.edges):
            raise ValueError("semantic edge references an unknown task")
        known_resources = set(resources)
        if any(
            effect.resource not in known_resources
            for task in self.tasks
            for effect in task.effects
        ):
            raise ValueError("semantic effect references an unknown state")
        for task in self.tasks:
            for task_input in task.inputs:
                match task_input.source:
                    case StateSelectionInputSource(resource=resource):
                        if resource not in known_resources:
                            raise ValueError(
                                "state selection references an unknown state"
                            )
                    case TaskResultInputSource(
                        task=producer,
                        output=output,
                    ):
                        producer_tasks = tuple(
                            candidate
                            for candidate in self.tasks
                            if candidate.symbol == producer
                        )
                        if len(producer_tasks) != 1 or all(
                            candidate.name != output
                            for candidate in producer_tasks[0].outputs
                        ):
                            raise ValueError(
                                "task result references an unknown producer output"
                            )
                        if SemanticEdge(producer, task.symbol) not in self.edges:
                            raise ValueError(
                                "task result requires an explicit dependency edge"
                            )
                    case _:
                        pass


@dataclass(frozen=True, slots=True)
class TimelineAnchor:
    symbol: AnchorSymbol
    ordinal: int

    def __post_init__(self) -> None:
        if not self.symbol or self.ordinal < 0:
            raise ValueError("timeline anchor has an invalid identity")


@dataclass(frozen=True, slots=True)
class SchedulerTimeline:
    anchors: tuple[TimelineAnchor, ...]
    capabilities: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not self.anchors
            or tuple(anchor.ordinal for anchor in self.anchors)
            != tuple(range(len(self.anchors)))
            or len({anchor.symbol for anchor in self.anchors}) != len(self.anchors)
            or tuple(sorted(self.capabilities)) != self.capabilities
            or len(set(self.capabilities)) != len(self.capabilities)
            or self.schema_version != 1
        ):
            raise ValueError("scheduler timeline is not canonical")


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

    def __post_init__(self) -> None:
        for value in (self.graph_digest, self.timeline_digest, self.signature):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("schedule decision digests must be SHA-256")
        if self.fallback_count < 0 or self.schema_version != 1:
            raise ValueError("invalid schedule decision envelope")


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

    def __post_init__(self) -> None:
        if (
            self.plan_epoch < 1
            or len(self.schedule_signature) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.schedule_signature
            )
            or self.fallback_count < 0
            or self.schema_version != 1
        ):
            raise ValueError("invalid canonical physical plan")


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
