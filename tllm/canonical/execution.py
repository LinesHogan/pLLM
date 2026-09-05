"""Strict execution for the first clean-room single-task vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from tllm.contracts.semantic_graph import AnchorId, ImplementationId, PhysicalPlan


InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
ExecutionInputT = TypeVar("ExecutionInputT")
ExecutionOutputT = TypeVar("ExecutionOutputT")
ExecutionContextT = TypeVar("ExecutionContextT")
TaskContextT = TypeVar("TaskContextT", contravariant=True)


class SingleTaskImplementation(Protocol[InputT, OutputT]):
    @property
    def implementation_id(self) -> ImplementationId: ...

    def execute(self, value: InputT) -> OutputT: ...


class TaskGraphImplementation(Protocol[TaskContextT]):
    @property
    def implementation_id(self) -> ImplementationId: ...

    def execute(self, context: TaskContextT) -> None: ...


@dataclass(frozen=True, slots=True)
class SingleTaskExecution(Generic[ExecutionInputT, ExecutionOutputT]):
    plan: PhysicalPlan
    implementation: SingleTaskImplementation[
        ExecutionInputT,
        ExecutionOutputT,
    ]

    def __post_init__(self) -> None:
        if len(self.plan.tasks) != 1 or self.plan.fallback_count != 0:
            raise ValueError(
                "single-task execution requires one zero-fallback physical task"
            )
        if (
            self.plan.tasks[0].implementation_id
            != self.implementation.implementation_id
        ):
            raise ValueError(
                "single-task implementation differs from the Rust physical plan"
            )

    def run(self, value: ExecutionInputT) -> ExecutionOutputT:
        return self.implementation.execute(value)


@dataclass(frozen=True, slots=True)
class TaskGraphExecution(Generic[ExecutionContextT]):
    """Mechanically execute Rust-planned tasks by dense semantic task ID."""

    plan: PhysicalPlan
    implementations: tuple[TaskGraphImplementation[ExecutionContextT], ...]

    def __post_init__(self) -> None:
        task_ids = tuple(int(task.task_id) for task in self.plan.tasks)
        if (
            self.plan.fallback_count != 0
            or len(task_ids) != len(self.implementations)
            or tuple(sorted(task_ids)) != tuple(range(len(task_ids)))
            or any(
                left.anchor_id > right.anchor_id
                for left, right in zip(
                    self.plan.tasks,
                    self.plan.tasks[1:],
                )
            )
        ):
            raise ValueError(
                "task-graph execution requires a complete zero-fallback dense plan"
            )
        if any(
            self.implementations[int(task.task_id)].implementation_id
            != task.implementation_id
            for task in self.plan.tasks
        ):
            raise ValueError(
                "task-graph implementation differs from the Rust physical plan"
            )

    def run(self, context: ExecutionContextT) -> ExecutionContextT:
        """Execute a synchronous invocation in the complete signed task order."""
        for task in self.plan.tasks:
            self.implementations[int(task.task_id)].execute(context)
        return context

    def start(
        self,
        context: ExecutionContextT,
    ) -> "TaskGraphInvocation[ExecutionContextT]":
        return TaskGraphInvocation(
            self.plan,
            context,
        )

    def execute_anchor(
        self,
        invocation: "TaskGraphInvocation[ExecutionContextT]",
        anchor_id: AnchorId,
    ) -> None:
        if invocation.plan is not self.plan:
            raise ValueError("task invocation belongs to another physical plan")
        invocation.execute_anchor(
            self.plan,
            self.implementations,
            anchor_id,
        )

    def finish(
        self,
        invocation: "TaskGraphInvocation[ExecutionContextT]",
    ) -> ExecutionContextT:
        if invocation.plan is not self.plan:
            raise ValueError("task invocation belongs to another physical plan")
        return invocation.finish(self.plan)


class TaskGraphInvocation(Generic[ExecutionContextT]):
    """One staged execution cursor advanced only by signed anchor events."""

    __slots__ = (
        "plan",
        "_context",
        "_next_task_index",
        "_last_anchor_id",
        "_finished",
    )

    def __init__(
        self,
        plan: PhysicalPlan,
        context: ExecutionContextT,
    ) -> None:
        self.plan = plan
        self._context = context
        self._next_task_index = 0
        self._last_anchor_id = -1
        self._finished = False

    @property
    def context(self) -> ExecutionContextT:
        return self._context

    @property
    def last_anchor_id(self) -> int:
        return self._last_anchor_id

    @property
    def finished(self) -> bool:
        return self._finished

    def execute_anchor(
        self,
        plan: PhysicalPlan,
        implementations: tuple[TaskGraphImplementation[ExecutionContextT], ...],
        anchor_id: AnchorId,
    ) -> None:
        dense_anchor = int(anchor_id)
        if self._finished or dense_anchor < 0 or dense_anchor < self._last_anchor_id:
            raise RuntimeError("task invocation anchor progression is invalid")
        if (
            self._next_task_index < len(plan.tasks)
            and int(plan.tasks[self._next_task_index].anchor_id) < dense_anchor
        ):
            raise RuntimeError("task invocation skipped a signed execution anchor")
        while (
            self._next_task_index < len(plan.tasks)
            and int(plan.tasks[self._next_task_index].anchor_id) == dense_anchor
        ):
            task = plan.tasks[self._next_task_index]
            implementations[int(task.task_id)].execute(self._context)
            self._next_task_index += 1
        self._last_anchor_id = dense_anchor

    def finish(self, plan: PhysicalPlan) -> ExecutionContextT:
        if self._finished or self._next_task_index != len(plan.tasks):
            raise RuntimeError("task invocation has unfinished signed tasks")
        self._finished = True
        return self._context


__all__ = [
    "SingleTaskExecution",
    "SingleTaskImplementation",
    "TaskGraphExecution",
    "TaskGraphImplementation",
    "TaskGraphInvocation",
]
