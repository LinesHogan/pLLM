"""Shape-only completion examples and host tensor accounting for learning authoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import TypeVar

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.feedback import Completion, CompletionBatch, FinishReason
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import SchedulerTimeline, TaskResourceFact, TaskSymbol, ValueResourceFact
from pllm.execution.declaration import Dimensions, Shape
from pllm.execution.host_runtime import CompletionInput, HostRuntimeFacts
from pllm.execution.memory import allocation_bytes, measure, optimizer_scratch
from pllm.kernels.adamw import AdamWConfig

ContextT = TypeVar("ContextT")


def completion_example(facts: HostRuntimeFacts, group_size: int = 1) -> CompletionInput:
    """A maximum-size fake CPU completion batch; no model or CUDA initialization."""
    rows = facts.max_input_rows
    if rows is None:
        raise ValueError("finite completion costs require max_input_rows")
    completions = CompletionBatch(
        tuple(
            Completion(
                RequestIdentity(str(i), i // group_size, i % group_size),
                i,
                str(i // group_size),
                (1,),
                r"\boxed{0}",
                FinishReason.STOP,
            )
            for i in range(rows)
        )
    )
    return CompletionInput(completions, torch.empty((rows, facts.hidden_size)))


def completion_resources(
    facts: HostRuntimeFacts,
    shapes: Mapping[str, Shape],
    dtype: torch.dtype,
    optimizer: AdamWConfig,
    example: Callable[[HostRuntimeFacts], ContextT],
    loss: Callable[[ContextT, Mapping[str, torch.Tensor]], torch.Tensor | None],
) -> SchedulerTimeline:
    """Derive completion forward/backward and optimizer scratch from its declared loss."""
    if facts.memory_budget_bytes is None:
        return facts.timeline()
    dimensions = Dimensions(facts.hidden_size, 0)
    with FakeTensorMode():
        data = example(facts)
        parameters = {
            name: torch.empty(shape(dimensions) if callable(shape) else shape, dtype=dtype, requires_grad=True)
            for name, shape in shapes.items()
        }

        def run() -> object:
            objective = loss(data, parameters)
            if objective is None:
                return ()
            derivatives = torch.autograd.grad(
                objective, tuple(parameters.values()), allow_unused=True, materialize_grads=True
            )
            return (objective, *derivatives)

        usage = measure(run)
    scratch = optimizer_scratch(tuple(parameters.values()), optimizer)
    rows = facts.max_input_rows
    assert rows is not None
    reserved = allocation_bytes(rows * facts.hidden_size * 4)
    return replace(
        facts.timeline(),
        reserved_bytes=(reserved,) * len(facts.timeline().anchors),
        task_resources=(
            TaskResourceFact(TaskSymbol("publish"), 0),
            TaskResourceFact(
                TaskSymbol("train"), usage.workspace + scratch.workspace, input_shapes=((rows, facts.hidden_size),)
            ),
        ),
        value_resources=(
            ValueResourceFact(TaskSymbol("publish"), "control", 0),
            ValueResourceFact(TaskSymbol("train"), "control", usage.output),
        ),
    )


def transaction_resources(
    facts: HostRuntimeFacts,
    example: Callable[[HostRuntimeFacts], tuple[ContextT, object]],
    functions: tuple[Callable[[ContextT], None], ...],
) -> SchedulerTimeline:
    """Trace a native host transaction with its rollback snapshot and retained context."""
    from pllm.execution.memory import tensors

    if facts.memory_budget_bytes is None:
        return facts.timeline()
    if facts.max_input_rows is None:
        raise ValueError("finite transaction costs require max_input_rows")
    tasks: list[TaskResourceFact] = []
    values: list[ValueResourceFact] = []
    with FakeTensorMode():
        context, snapshot = example(facts)
        resident = {
            value.untyped_storage(): allocation_bytes(value.untyped_storage().nbytes())
            for value in tensors((context, snapshot))
        }
        for name, output, function in zip(
            ("update", "query", "publish"), ("context", "context", "result"), functions, strict=True
        ):

            def run() -> object:
                function(context)
                return tuple(value for value in tensors(context) if value.untyped_storage() not in resident)

            usage = measure(run)
            tasks.append(
                TaskResourceFact(
                    TaskSymbol(name), usage.workspace, input_shapes=((facts.max_input_rows, facts.hidden_size),)
                )
            )
            values.append(ValueResourceFact(TaskSymbol(name), output, usage.output))
    timeline = facts.timeline()
    return replace(
        timeline,
        reserved_bytes=(sum(resident.values()),) * len(timeline.anchors),
        task_resources=tuple(sorted(tasks, key=lambda fact: fact.task)),
        value_resources=tuple(sorted(values, key=lambda fact: fact.task)),
    )
