"""Rust planning facts shared by workload execution and backend attachment."""

from __future__ import annotations

from dataclasses import dataclass

from pllm.contracts.semantic_graph import (
    ScheduleDecision,
    SchedulerTimeline,
    SemanticGraph,
)
from pllm.contracts.state_representation import (
    StatePhysicalPlan,
    StateRepresentationDecision,
    StateRepresentationRequest,
)
from pllm.contracts.tensor_layout import TensorLayoutDecision, TensorLayoutRequest


@dataclass(frozen=True, slots=True)
class PlanBundle:
    """The complete reusable plan facts for one specialization."""

    semantic_graph: SemanticGraph
    timeline: SchedulerTimeline
    schedule: ScheduleDecision
    state_request: StateRepresentationRequest
    state_decision: StateRepresentationDecision
    state_physical: StatePhysicalPlan
    tensor_layout_request: TensorLayoutRequest
    tensor_layout_decision: TensorLayoutDecision


__all__ = ["PlanBundle"]
