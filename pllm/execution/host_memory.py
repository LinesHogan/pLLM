"""Trace the executed host transaction and the context it retains between operations."""

from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.semantic_graph import SchedulerTimeline, TaskResourceFact, TaskSymbol, ValueResourceFact
from pllm.execution.declaration import HostOperation
from pllm.execution.host_runtime import HostRuntimeFacts
from pllm.execution.memory import MemoryTrace

ContextT = TypeVar("ContextT")


def transaction_resources(
    facts: HostRuntimeFacts,
    example: Callable[[HostRuntimeFacts], tuple[ContextT, object]],
    operations: tuple[HostOperation[ContextT], ...],
) -> SchedulerTimeline:
    """Execute the same transaction functions, with its live context and rollback snapshot."""
    timeline = facts.timeline()
    if facts.memory_budget_bytes is None:
        return timeline
    if facts.max_input_rows is None:
        raise ValueError("finite transaction costs require max_input_rows")
    tasks: list[TaskResourceFact] = []
    values: list[ValueResourceFact] = []
    with FakeTensorMode(), MemoryTrace() as lifetime:
        context, snapshot = example(facts)
        resident = lifetime.live
        for name, output, operation in zip(
            ("update", "query", "publish"), ("context", "context", "result"), operations, strict=True
        ):
            usage = operation.memory(context)
            tasks.append(
                TaskResourceFact(
                    TaskSymbol(name), usage.workspace, input_shapes=((facts.max_input_rows, facts.hidden_size),)
                )
            )
            values.append(ValueResourceFact(TaskSymbol(name), output, max(0, lifetime.live - resident)))
        del snapshot
    return replace(
        timeline,
        reserved_bytes=(resident,) * len(timeline.anchors),
        task_resources=tuple(sorted(tasks, key=lambda item: item.task)),
        value_resources=tuple(sorted(values, key=lambda item: item.task)),
    )
