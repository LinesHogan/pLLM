"""Closed FFI for clean-room semantic scheduling and physical lowering."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import cast

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


def _json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    )


def _graph_json(graph: SemanticGraph) -> str:
    # The dataclasses are the schema. Only the closed union tag is wire-specific.
    wire = asdict(graph)
    for task, task_wire in zip(graph.tasks, wire["tasks"], strict=True):
        for task_input, input_wire in zip(task.inputs, task_wire["inputs"], strict=True):
            match task_input.source:
                case HiddenInputSource():
                    kind = "hidden"
                case StepIdInputSource():
                    kind = "step_id"
                case PackedRowsInputSource():
                    kind = "packed_rows"
                case PackedSamplerRowsInputSource():
                    kind = "packed_sampler_rows"
                case HostServiceInputSource():
                    kind = "host_service"
                case StateSelectionInputSource():
                    kind = "state_selection"
                case TaskResultInputSource():
                    kind = "result"
                case _:
                    raise TypeError("semantic input source is outside the closed codec")
            input_wire["source"]["kind"] = kind
    return _json(wire)


def _timeline_json(timeline: SchedulerTimeline) -> str:
    return _json(asdict(timeline))


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
    return _json(asdict(decision))


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
    return _json(asdict(plan))


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
