"""Closed, Rust-signed plans produced by one canonical workload compilation."""

from __future__ import annotations

from dataclasses import dataclass

from pllm.contracts.semantic_graph import (
    PhysicalPlan,
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
    """The complete reusable signed plan facts for one specialization."""

    semantic_graph: SemanticGraph
    timeline: SchedulerTimeline
    schedule_decision: ScheduleDecision
    schedule_physical: PhysicalPlan
    state_request: StateRepresentationRequest
    state_decision: StateRepresentationDecision
    state_physical: StatePhysicalPlan
    tensor_layout_request: TensorLayoutRequest
    tensor_layout_decision: TensorLayoutDecision


__all__ = ["PlanBundle"]
