"""Mechanical execution of Rust physical tasks and value releases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pllm.contracts.semantic_graph import ImplementationId, ScheduleDecision
from pllm.execution.trace import ExecutionTrace

ExecutionContextT = TypeVar("ExecutionContextT")
TaskContextT = TypeVar("TaskContextT", contravariant=True)
ExecutionResultT = TypeVar("ExecutionResultT")
TaskResultT = TypeVar("TaskResultT", covariant=True)


class TaskGraphImplementation(Protocol[TaskContextT, TaskResultT]):
    @property
    def implementation_id(self) -> ImplementationId: ...

    def execute(self, context: TaskContextT) -> TaskResultT: ...


@dataclass(slots=True)
class TaskGraphExecution(Generic[ExecutionContextT, ExecutionResultT]):
    """Mechanically execute Rust-planned tasks by dense semantic task ID."""

    plan: ScheduleDecision
    implementations: tuple[TaskGraphImplementation[ExecutionContextT, ExecutionResultT], ...]
    trace: ExecutionTrace
    release_values: Callable[[ExecutionContextT, tuple[int, ...]], None] | None = None

    def __post_init__(self) -> None:
        bindings = {implementation.implementation_id: implementation for implementation in self.implementations}
        assert len(bindings) == len(self.implementations)
        assert set(bindings) == {task.implementation_id for task in self.plan.tasks}
        assert self.trace.plan is self.plan
        # Rust numbers tasks densely, so position i holds the implementation of task i.
        self.implementations = tuple(
            bindings[task.implementation_id] for task in sorted(self.plan.tasks, key=lambda task: task.task_id)
        )

    def run(self, context: ExecutionContextT, invocation_id: int) -> ExecutionContextT:
        """Run every planned host task synchronously and return its context."""
        for task in self.plan.tasks:
            with self.trace.task(invocation_id, task.task_id):
                self.implementations[int(task.task_id)].execute(context)
                if task.release_values and self.release_values is not None:
                    self.release_values(context, task.release_values)
        return context


__all__ = [
    "TaskGraphExecution",
    "TaskGraphImplementation",
]
