"""Mechanical execution of original Rust physical tasks and region boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import torch

from pllm.contracts.semantic_graph import AnchorId, ImplementationId, PhysicalPlan, TaskId
from pllm.execution.trace import ExecutionTrace

ExecutionContextT = TypeVar("ExecutionContextT")
TaskContextT = TypeVar("TaskContextT", contravariant=True)


class TaskGraphImplementation(Protocol[TaskContextT]):
    @property
    def implementation_id(self) -> ImplementationId: ...

    def execute(self, context: TaskContextT) -> None: ...


@dataclass(slots=True)
class TaskGraphExecution(Generic[ExecutionContextT]):
    """Mechanically execute Rust-planned tasks by dense semantic task ID."""

    plan: PhysicalPlan
    implementations: tuple[TaskGraphImplementation[ExecutionContextT], ...]
    trace: ExecutionTrace
    release_values: Callable[[ExecutionContextT, tuple[int, ...]], None] | None = None

    def __post_init__(self) -> None:
        bindings = {implementation.implementation_id: implementation for implementation in self.implementations}
        if len(bindings) != len(self.implementations) or set(bindings) != {
            task.implementation_id for task in self.plan.tasks
        }:
            raise ValueError("native bindings must name exactly the Rust-selected implementations")
        self.implementations = tuple(
            bindings[task.implementation_id] for task in sorted(self.plan.tasks, key=lambda task: task.task_id)
        )
        if (
            self.trace.plan is not self.plan
            or len(self.plan.tasks) != len(self.implementations)
            or any(
                self.implementations[int(task.task_id)].implementation_id != task.implementation_id
                for task in self.plan.tasks
            )
        ):
            raise ValueError("task-graph implementation differs from the Rust physical plan")


class TaskGraphInvocation(Generic[ExecutionContextT]):
    """One cursor for synchronous host and externally staged execution."""

    def __init__(
        self, execution: TaskGraphExecution[ExecutionContextT], context: ExecutionContextT, invocation_id: int
    ) -> None:
        self.execution, self.context, self.invocation_id = execution, context, invocation_id
        self._next_task_index = 0
        self.last_anchor_id = -1
        self.finished = False

    def run(self) -> ExecutionContextT:
        for anchor in dict.fromkeys(task.anchor_id for task in self.execution.plan.tasks):
            self.execute_anchor(anchor)
        return self.finish()

    def execute_anchor(self, anchor_id: AnchorId, stream: torch.cuda.Stream | None = None) -> None:
        plan, trace = self.execution.plan, self.execution.trace
        dense_anchor = int(anchor_id)
        if self.finished or dense_anchor < 0 or dense_anchor < self.last_anchor_id:
            raise RuntimeError("task invocation anchor progression is invalid")
        if self._next_task_index < len(plan.tasks) and int(plan.tasks[self._next_task_index].anchor_id) < dense_anchor:
            raise RuntimeError("task invocation skipped a signed execution anchor")
        while (
            self._next_task_index < len(plan.tasks) and int(plan.tasks[self._next_task_index].anchor_id) == dense_anchor
        ):
            end = next(end for end in plan.region_ends if end > self._next_task_index)
            while self._next_task_index < end:
                task = plan.tasks[self._next_task_index]
                with trace.task(self.invocation_id, task.task_id, stream):
                    self.execution.implementations[int(task.task_id)].execute(self.context)
                    if task.release_values and self.execution.release_values is not None:
                        self.execution.release_values(self.context, task.release_values)
                self._next_task_index += 1
        self.last_anchor_id = dense_anchor

    def finish(self) -> ExecutionContextT:
        if self.finished or self._next_task_index != len(self.execution.plan.tasks):
            raise RuntimeError("task invocation has unfinished signed tasks")
        self.finished = True
        return self.context

    def execute_part(self, task_id: TaskId, *, complete: bool, stream: torch.cuda.Stream | None) -> None:
        """Run a projection segment of the current task, retaining its cursor until completion."""
        plan, trace = self.execution.plan, self.execution.trace
        assert not self.finished and self._next_task_index < len(plan.tasks)
        task = plan.tasks[self._next_task_index]
        assert task.task_id == task_id and task.anchor_id >= self.last_anchor_id
        with trace.task(self.invocation_id, task_id, stream):
            self.execution.implementations[int(task_id)].execute(self.context)
        self.last_anchor_id = int(task.anchor_id)
        if complete:
            if task.release_values and self.execution.release_values is not None:
                self.execution.release_values(self.context, task.release_values)
            self._next_task_index += 1


__all__ = [
    "TaskGraphExecution",
    "TaskGraphImplementation",
    "TaskGraphInvocation",
]
