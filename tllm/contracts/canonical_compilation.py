"""Closed, Rust-signed plans produced by one canonical workload compilation."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.contracts.semantic_graph import (
    PhysicalPlan,
    ScheduleDecision,
    SchedulerTimeline,
    SemanticGraph,
)
from tllm.contracts.state_representation import (
    StatePhysicalPlan,
    StateRepresentationDecision,
    StateRepresentationRequest,
)


@dataclass(frozen=True, slots=True)
class CanonicalScheduleBundle:
    """Complete Rust-signed schedule facts for non-represented instance state."""

    semantic_graph: SemanticGraph
    timeline: SchedulerTimeline
    schedule_decision: ScheduleDecision
    schedule_physical: PhysicalPlan

    def __post_init__(self) -> None:
        if (
            type(self.semantic_graph) is not SemanticGraph
            or type(self.timeline) is not SchedulerTimeline
            or type(self.schedule_decision) is not ScheduleDecision
            or type(self.schedule_physical) is not PhysicalPlan
        ):
            raise TypeError("canonical schedule bundle requires exact signed facts")


@dataclass(frozen=True, slots=True)
class CanonicalPlanBundle:
    """The complete reusable signed plan facts for one specialization."""

    semantic_graph: SemanticGraph
    timeline: SchedulerTimeline
    schedule_decision: ScheduleDecision
    schedule_physical: PhysicalPlan
    state_request: StateRepresentationRequest
    state_decision: StateRepresentationDecision
    state_physical: StatePhysicalPlan

    def __post_init__(self) -> None:
        if (
            type(self.semantic_graph) is not SemanticGraph
            or type(self.timeline) is not SchedulerTimeline
            or type(self.schedule_decision) is not ScheduleDecision
            or type(self.schedule_physical) is not PhysicalPlan
            or type(self.state_request) is not StateRepresentationRequest
            or type(self.state_decision) is not StateRepresentationDecision
            or type(self.state_physical) is not StatePhysicalPlan
        ):
            raise TypeError("canonical plan bundle requires exact signed plan facts")


__all__ = ["CanonicalPlanBundle", "CanonicalScheduleBundle"]
