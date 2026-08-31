"""Clean-room semantic graph, schedule, and physical-plan contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite


WorkloadId = NewType("WorkloadId", str)
TaskSymbol = NewType("TaskSymbol", str)
ImplementationId = NewType("ImplementationId", str)
AnchorSymbol = NewType("AnchorSymbol", str)
TaskId = NewType("TaskId", int)
AnchorId = NewType("AnchorId", int)


class ExecutionKind(StrEnum):
    HOST_SERVICE = "host_service"


@dataclass(frozen=True, slots=True)
class HiddenInputSource:
    layer: int
    site: HiddenSite
    phase: HiddenPhase


@dataclass(frozen=True, slots=True)
class StepIdInputSource:
    pass


SemanticInputSource = HiddenInputSource | StepIdInputSource


@dataclass(frozen=True, slots=True)
class SemanticInput:
    name: str
    source: SemanticInputSource

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("semantic input requires a name")


@dataclass(frozen=True, slots=True)
class SemanticOutput:
    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("semantic output requires a name")


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
        if not self.implementation_id:
            raise ValueError("task implementation requires a stable ID")
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

    def __post_init__(self) -> None:
        if not self.symbol or not self.implementations:
            raise ValueError("semantic task requires a symbol and implementations")
        input_names = tuple(item.name for item in self.inputs)
        output_names = tuple(item.name for item in self.outputs)
        implementation_ids = tuple(
            item.implementation_id for item in self.implementations
        )
        if (
            len(set(input_names)) != len(input_names)
            or len(set(output_names)) != len(output_names)
            or tuple(sorted(implementation_ids)) != implementation_ids
            or len(set(implementation_ids)) != len(implementation_ids)
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
    edges: tuple[SemanticEdge, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        symbols = tuple(task.symbol for task in self.tasks)
        edges = tuple((edge.source, edge.target) for edge in self.edges)
        if (
            not self.workload_id
            or not symbols
            or tuple(sorted(symbols)) != symbols
            or len(set(symbols)) != len(symbols)
            or tuple(sorted(edges)) != edges
            or len(set(edges)) != len(edges)
            or self.schema_version != 1
        ):
            raise ValueError("semantic graph is not canonical")
        known = set(symbols)
        if any(edge.source not in known or edge.target not in known for edge in self.edges):
            raise ValueError("semantic edge references an unknown task")


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
    "ImplementationId",
    "PhysicalPlan",
    "PhysicalTask",
    "ScheduleDecision",
    "ScheduledTask",
    "SchedulerTimeline",
    "SemanticEdge",
    "SemanticGraph",
    "SemanticInput",
    "SemanticOutput",
    "SemanticTask",
    "StepIdInputSource",
    "TaskId",
    "TaskImplementationSpec",
    "TaskSymbol",
    "TimelineAnchor",
    "TimingConstraint",
    "WorkloadId",
]
