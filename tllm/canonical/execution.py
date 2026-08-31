"""Strict execution for the first clean-room single-task vertical slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from tllm.contracts.semantic_graph import ImplementationId, PhysicalPlan


InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
ExecutionInputT = TypeVar("ExecutionInputT")
ExecutionOutputT = TypeVar("ExecutionOutputT")


class SingleTaskImplementation(Protocol[InputT, OutputT]):
    @property
    def implementation_id(self) -> ImplementationId: ...

    def execute(self, value: InputT) -> OutputT: ...


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


__all__ = ["SingleTaskExecution", "SingleTaskImplementation"]
