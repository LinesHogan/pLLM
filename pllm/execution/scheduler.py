"""Rust scheduling and layout interning for workload compilation."""

from __future__ import annotations

from pllm import _core
from pllm.contracts.semantic_graph import ScheduleDecision, SchedulerTimeline, SemanticGraph
from pllm.contracts.state_representation import (
    StateRepresentationDecision,
    StateRepresentationRequest,
)
from pllm.contracts.tensor_layout import TensorLayoutDecision, TensorLayoutRequest
from pllm.execution.codec import decode_boundary, encode_boundary


def compile_state_schedule(
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    request: StateRepresentationRequest,
) -> tuple[StateRepresentationDecision, ScheduleDecision]:
    decision, schedule = _core.compile_state_schedule(
        encode_boundary(graph), encode_boundary(timeline), encode_boundary(request)
    )
    return (
        decode_boundary(decision, StateRepresentationDecision),
        decode_boundary(schedule, ScheduleDecision),
    )


def compile_schedule(
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    *,
    state: StateRepresentationDecision | None = None,
) -> ScheduleDecision:
    return decode_boundary(
        _core.compile_schedule(
            encode_boundary(graph),
            encode_boundary(timeline),
            None if state is None else encode_boundary(state),
        ),
        ScheduleDecision,
    )


def intern_tensor_layouts(request: TensorLayoutRequest) -> TensorLayoutDecision:
    return decode_boundary(_core.intern_tensor_layouts(encode_boundary(request)), TensorLayoutDecision)


__all__ = [
    "compile_state_schedule",
    "intern_tensor_layouts",
    "compile_schedule",
]
