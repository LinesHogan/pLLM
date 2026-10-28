"""Closed FFI for clean-room semantic scheduling, layout interning and physical lowering."""

from __future__ import annotations

from pllm import _core
from pllm.contracts.semantic_graph import PhysicalPlan, ScheduleDecision, SchedulerTimeline, SemanticGraph
from pllm.contracts.state_representation import (
    StatePhysicalPlan,
    StateRepresentationDecision,
    StateRepresentationRequest,
)
from pllm.contracts.tensor_layout import TensorLayoutDecision, TensorLayoutRequest
from pllm.execution.codec import decode_boundary, encode_boundary


def compile_state_schedule(
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    request: StateRepresentationRequest,
    *,
    plan_epoch: int = 1,
) -> tuple[StateRepresentationDecision, StatePhysicalPlan, ScheduleDecision, PhysicalPlan]:
    decision, state, schedule, physical = _core.compile_state_schedule(
        encode_boundary(graph), encode_boundary(timeline), encode_boundary(request), plan_epoch
    )
    return (
        decode_boundary(decision, StateRepresentationDecision),
        decode_boundary(state, StatePhysicalPlan),
        decode_boundary(schedule, ScheduleDecision),
        decode_boundary(physical, PhysicalPlan),
    )


def compile_schedule(
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    *,
    state: StatePhysicalPlan | None = None,
) -> ScheduleDecision:
    return decode_boundary(
        _core.compile_schedule(
            encode_boundary(graph),
            encode_boundary(timeline),
            None if state is None else encode_boundary(state),
        ),
        ScheduleDecision,
    )


def lower_physical_plan(
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    decision: ScheduleDecision,
    *,
    plan_epoch: int,
    state: StatePhysicalPlan | None = None,
) -> PhysicalPlan:
    return decode_boundary(
        _core.lower_physical_plan(
            encode_boundary(graph),
            encode_boundary(timeline),
            encode_boundary(decision),
            plan_epoch,
            None if state is None else encode_boundary(state),
        ),
        PhysicalPlan,
    )


def intern_tensor_layouts(request: TensorLayoutRequest) -> TensorLayoutDecision:
    return decode_boundary(_core.intern_tensor_layouts(encode_boundary(request)), TensorLayoutDecision)


__all__ = [
    "compile_state_schedule",
    "intern_tensor_layouts",
    "compile_schedule",
    "lower_physical_plan",
]
