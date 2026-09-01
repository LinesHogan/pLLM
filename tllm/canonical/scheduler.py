"""Closed FFI for clean-room semantic scheduling and physical lowering."""

from __future__ import annotations

import json
from typing import Literal, TypedDict, cast

from tllm import _core
from tllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    HiddenInputSource,
    HostServiceInputSource,
    ImplementationId,
    PhysicalPlan,
    PhysicalTask,
    PackedRowsInputSource,
    PackedSamplerRowsInputSource,
    ScheduleDecision,
    ScheduledTask,
    SchedulerTimeline,
    SemanticGraph,
    StateSelectionInputSource,
    StepIdInputSource,
    TaskId,
    TaskSymbol,
    TaskResultInputSource,
)


class _TimingWire(TypedDict):
    after: str
    before: str


class _HiddenSourceWire(TypedDict):
    kind: Literal["hidden"]
    layer: int
    site: str
    phase: str


class _StepIdSourceWire(TypedDict):
    kind: Literal["step_id"]


class _PackedRowsSourceWire(TypedDict):
    kind: Literal["packed_rows"]


class _PackedSamplerRowsSourceWire(TypedDict):
    kind: Literal["packed_sampler_rows"]


class _HostServiceSourceWire(TypedDict):
    kind: Literal["host_service"]
    service: str
    port: str
    schema: str


class _StateSelectionSourceWire(TypedDict):
    kind: Literal["state_selection"]
    resource: str
    phase: str


class _TaskResultSourceWire(TypedDict):
    kind: Literal["result"]
    task: str
    output: str


_InputSourceWire = (
    _HiddenSourceWire
    | _StepIdSourceWire
    | _PackedRowsSourceWire
    | _PackedSamplerRowsSourceWire
    | _HostServiceSourceWire
    | _StateSelectionSourceWire
    | _TaskResultSourceWire
)


class _InputWire(TypedDict):
    name: str
    source: _InputSourceWire


class _OutputWire(TypedDict):
    name: str
    kind: str
    schema: str


class _ImplementationWire(TypedDict):
    implementation_id: str
    execution_kind: str
    required_capabilities: list[str]


class _EffectWire(TypedDict):
    resource: str
    kind: str


class _TaskWire(TypedDict):
    symbol: str
    timing: _TimingWire
    inputs: list[_InputWire]
    outputs: list[_OutputWire]
    implementations: list[_ImplementationWire]
    effects: list[_EffectWire]


class _EdgeWire(TypedDict):
    source: str
    target: str


class _StateWire(TypedDict):
    resource: str
    scope: str
    key_fields: list[str]


class _GraphWire(TypedDict):
    workload_id: str
    tasks: list[_TaskWire]
    execution_parameters_digest: str
    states: list[_StateWire]
    edges: list[_EdgeWire]
    schema_version: int


class _AnchorWire(TypedDict):
    symbol: str
    ordinal: int


class _TimelineWire(TypedDict):
    anchors: list[_AnchorWire]
    capabilities: list[str]
    schema_version: int


def _graph_json(graph: SemanticGraph) -> str:
    tasks: list[_TaskWire] = []
    for task in graph.tasks:
        inputs: list[_InputWire] = []
        for task_input in task.inputs:
            match task_input.source:
                case HiddenInputSource(layer=layer, site=site, phase=phase):
                    inputs.append(
                        _InputWire(
                            name=task_input.name,
                            source=_HiddenSourceWire(
                                kind="hidden",
                                layer=layer,
                                site=site.value,
                                phase=phase.value,
                            ),
                        )
                    )
                case StepIdInputSource():
                    inputs.append(
                        _InputWire(
                            name=task_input.name,
                            source=_StepIdSourceWire(kind="step_id"),
                        )
                    )
                case PackedRowsInputSource():
                    inputs.append(
                        _InputWire(
                            name=task_input.name,
                            source=_PackedRowsSourceWire(kind="packed_rows"),
                        )
                    )
                case PackedSamplerRowsInputSource():
                    inputs.append(
                        _InputWire(
                            name=task_input.name,
                            source=_PackedSamplerRowsSourceWire(
                                kind="packed_sampler_rows"
                            ),
                        )
                    )
                case HostServiceInputSource(
                    service=service,
                    port=port,
                    schema=schema,
                ):
                    inputs.append(
                        _InputWire(
                            name=task_input.name,
                            source=_HostServiceSourceWire(
                                kind="host_service",
                                service=str(service),
                                port=str(port),
                                schema=str(schema),
                            ),
                        )
                    )
                case StateSelectionInputSource(
                    resource=resource,
                    phase=phase,
                ):
                    inputs.append(
                        _InputWire(
                            name=task_input.name,
                            source=_StateSelectionSourceWire(
                                kind="state_selection",
                                resource=str(resource),
                                phase=phase.value,
                            ),
                        )
                    )
                case TaskResultInputSource(task=producer, output=output):
                    inputs.append(
                        _InputWire(
                            name=task_input.name,
                            source=_TaskResultSourceWire(
                                kind="result",
                                task=str(producer),
                                output=output,
                            ),
                        )
                    )
        tasks.append(
            _TaskWire(
                symbol=str(task.symbol),
                timing=_TimingWire(
                    after=str(task.timing.after),
                    before=str(task.timing.before),
                ),
                inputs=inputs,
                outputs=[
                    _OutputWire(
                        name=output.name,
                        kind=output.kind.value,
                        schema=str(output.schema),
                    )
                    for output in task.outputs
                ],
                implementations=[
                    _ImplementationWire(
                        implementation_id=str(implementation.implementation_id),
                        execution_kind=implementation.execution_kind.value,
                        required_capabilities=list(
                            implementation.required_capabilities
                        ),
                    )
                    for implementation in task.implementations
                ],
                effects=[
                    _EffectWire(
                        resource=str(effect.resource),
                        kind=effect.kind.value,
                    )
                    for effect in task.effects
                ],
            )
        )
    wire = _GraphWire(
        workload_id=str(graph.workload_id),
        tasks=tasks,
        execution_parameters_digest=graph.execution_parameters_digest,
        states=[
            _StateWire(
                resource=str(state.resource),
                scope=state.scope.value,
                key_fields=[field.value for field in state.key_fields],
            )
            for state in graph.states
        ],
        edges=[
            _EdgeWire(source=str(edge.source), target=str(edge.target))
            for edge in graph.edges
        ],
        schema_version=graph.schema_version,
    )
    return json.dumps(
        wire,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _timeline_json(timeline: SchedulerTimeline) -> str:
    wire = _TimelineWire(
        anchors=[
            _AnchorWire(symbol=str(anchor.symbol), ordinal=anchor.ordinal)
            for anchor in timeline.anchors
        ],
        capabilities=list(timeline.capabilities),
        schema_version=timeline.schema_version,
    )
    return json.dumps(
        wire,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _scheduled_task(raw: object, label: str) -> ScheduledTask:
    if type(raw) is not dict:
        raise TypeError(f"{label} must be an object")
    fields = cast(dict[object, object], raw)
    if set(fields) != {
        "task_id",
        "task_symbol",
        "implementation_id",
        "anchor_id",
        "anchor_symbol",
    }:
        raise ValueError(f"{label} has the wrong fields")
    task_id = fields["task_id"]
    task_symbol = fields["task_symbol"]
    implementation_id = fields["implementation_id"]
    anchor_id = fields["anchor_id"]
    anchor_symbol = fields["anchor_symbol"]
    if (
        type(task_id) is not int
        or type(task_symbol) is not str
        or type(implementation_id) is not str
        or type(anchor_id) is not int
        or type(anchor_symbol) is not str
    ):
        raise TypeError(f"{label} has invalid field types")
    return ScheduledTask(
        task_id=TaskId(task_id),
        task_symbol=TaskSymbol(task_symbol),
        implementation_id=ImplementationId(implementation_id),
        anchor_id=AnchorId(anchor_id),
        anchor_symbol=AnchorSymbol(anchor_symbol),
    )


def _decision_from_boundary(payload: str) -> ScheduleDecision:
    decoded = cast(object, json.loads(payload))
    if type(decoded) is not dict:
        raise TypeError("canonical schedule decision must be an object")
    fields = cast(dict[object, object], decoded)
    if set(fields) != {
        "graph_digest",
        "timeline_digest",
        "tasks",
        "fallback_count",
        "signature",
        "schema_version",
    }:
        raise ValueError("canonical schedule decision has the wrong fields")
    graph_digest = fields["graph_digest"]
    timeline_digest = fields["timeline_digest"]
    tasks = fields["tasks"]
    fallback_count = fields["fallback_count"]
    signature = fields["signature"]
    schema_version = fields["schema_version"]
    if (
        type(graph_digest) is not str
        or type(timeline_digest) is not str
        or type(tasks) is not list
        or type(fallback_count) is not int
        or type(signature) is not str
        or type(schema_version) is not int
    ):
        raise TypeError("canonical schedule decision has invalid field types")
    return ScheduleDecision(
        graph_digest=graph_digest,
        timeline_digest=timeline_digest,
        tasks=tuple(
            _scheduled_task(task, f"tasks[{position}]")
            for position, task in enumerate(cast(list[object], tasks))
        ),
        fallback_count=fallback_count,
        signature=signature,
        schema_version=schema_version,
    )


def _decision_json(decision: ScheduleDecision) -> str:
    return json.dumps(
        {
            "graph_digest": decision.graph_digest,
            "timeline_digest": decision.timeline_digest,
            "tasks": [
                {
                    "task_id": int(task.task_id),
                    "task_symbol": str(task.task_symbol),
                    "implementation_id": str(task.implementation_id),
                    "anchor_id": int(task.anchor_id),
                    "anchor_symbol": str(task.anchor_symbol),
                }
                for task in decision.tasks
            ],
            "fallback_count": decision.fallback_count,
            "signature": decision.signature,
            "schema_version": decision.schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _physical_from_boundary(payload: str) -> PhysicalPlan:
    decoded = cast(object, json.loads(payload))
    if type(decoded) is not dict:
        raise TypeError("canonical physical plan must be an object")
    fields = cast(dict[object, object], decoded)
    if set(fields) != {
        "plan_epoch",
        "schedule_signature",
        "tasks",
        "fallback_count",
        "schema_version",
    }:
        raise ValueError("canonical physical plan has the wrong fields")
    plan_epoch = fields["plan_epoch"]
    schedule_signature = fields["schedule_signature"]
    tasks = fields["tasks"]
    fallback_count = fields["fallback_count"]
    schema_version = fields["schema_version"]
    if (
        type(plan_epoch) is not int
        or type(schedule_signature) is not str
        or type(tasks) is not list
        or type(fallback_count) is not int
        or type(schema_version) is not int
    ):
        raise TypeError("canonical physical plan has invalid field types")
    parsed_tasks: list[PhysicalTask] = []
    for position, task in enumerate(cast(list[object], tasks)):
        if type(task) is not dict:
            raise TypeError(f"physical tasks[{position}] must be an object")
        task_fields = cast(dict[object, object], task)
        if set(task_fields) != {"task_id", "implementation_id", "anchor_id"}:
            raise ValueError(f"physical tasks[{position}] has the wrong fields")
        task_id = task_fields["task_id"]
        implementation_id = task_fields["implementation_id"]
        anchor_id = task_fields["anchor_id"]
        if (
            type(task_id) is not int
            or type(implementation_id) is not str
            or type(anchor_id) is not int
        ):
            raise TypeError(f"physical tasks[{position}] has invalid field types")
        parsed_tasks.append(
            PhysicalTask(
                task_id=TaskId(task_id),
                implementation_id=ImplementationId(implementation_id),
                anchor_id=AnchorId(anchor_id),
            )
        )
    return PhysicalPlan(
        plan_epoch=plan_epoch,
        schedule_signature=schedule_signature,
        tasks=tuple(parsed_tasks),
        fallback_count=fallback_count,
        schema_version=schema_version,
    )


def _physical_json(plan: PhysicalPlan) -> str:
    return json.dumps(
        {
            "plan_epoch": plan.plan_epoch,
            "schedule_signature": plan.schedule_signature,
            "tasks": [
                {
                    "task_id": int(task.task_id),
                    "implementation_id": str(task.implementation_id),
                    "anchor_id": int(task.anchor_id),
                }
                for task in plan.tasks
            ],
            "fallback_count": plan.fallback_count,
            "schema_version": plan.schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def compile_schedule(
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
) -> ScheduleDecision:
    return _decision_from_boundary(
        _core.compile_canonical_schedule(
            _graph_json(graph),
            _timeline_json(timeline),
        )
    )


def validate_schedule(
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    decision: ScheduleDecision,
) -> None:
    _core.validate_canonical_schedule(
        _graph_json(graph),
        _timeline_json(timeline),
        _decision_json(decision),
    )


def lower_physical_plan(
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    decision: ScheduleDecision,
    *,
    plan_epoch: int,
) -> PhysicalPlan:
    return _physical_from_boundary(
        _core.lower_canonical_physical_plan(
            _graph_json(graph),
            _timeline_json(timeline),
            _decision_json(decision),
            plan_epoch,
        )
    )


def validate_physical_plan(
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    decision: ScheduleDecision,
    plan: PhysicalPlan,
) -> None:
    _core.validate_canonical_physical_plan(
        _graph_json(graph),
        _timeline_json(timeline),
        _decision_json(decision),
        _physical_json(plan),
    )


encode_semantic_graph = _graph_json
encode_scheduler_timeline = _timeline_json
encode_schedule_decision = _decision_json
encode_schedule_physical_plan = _physical_json


__all__ = [
    "compile_schedule",
    "encode_schedule_decision",
    "encode_schedule_physical_plan",
    "encode_scheduler_timeline",
    "encode_semantic_graph",
    "lower_physical_plan",
    "validate_physical_plan",
    "validate_schedule",
]
