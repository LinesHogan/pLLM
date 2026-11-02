"""Rust planning facts shared by workload execution and backend attachment."""

from __future__ import annotations

from dataclasses import dataclass

from pllm.contracts.semantic_graph import (
    PhysicalTask,
    ScheduleDecision,
    SchedulerTimeline,
    SemanticGraph,
    TaskSymbol,
)
from pllm.contracts.state_representation import StateRepresentationDecision
from pllm.contracts.tensor_layout import TensorLayoutDecision, TensorLayoutRequest


@dataclass(frozen=True, slots=True)
class PlanBundle:
    """The complete reusable plan facts for one specialization."""

    semantic_graph: SemanticGraph
    timeline: SchedulerTimeline
    schedule: ScheduleDecision
    states: StateRepresentationDecision
    tensor_layout_request: TensorLayoutRequest
    tensor_layout_decision: TensorLayoutDecision

    def scheduled(self, symbol: TaskSymbol) -> PhysicalTask:
        """Where the plan placed the declared task `symbol`."""
        return next(item for item in self.schedule.tasks if self.semantic_graph.tasks[item.task_id].symbol == symbol)


__all__ = ["PlanBundle"]
