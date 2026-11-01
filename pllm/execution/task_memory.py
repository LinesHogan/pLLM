"""Memory planning for bound tensor operations.

Builds maximum-shape meta inputs for every operation (example), traces each operation's workspace and capture
residency (derive_resources, derive_host_resources) and orders producers before their readers. Workload
specialization (usage.py) calls it before planning.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.auxiliary import WeightedCEBatch, WeightedRowBatch
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_hidden import HiddenRowBatch, HiddenRows
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.parameter_overlay import CapturedParameterOperand
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import SamplerFeedback
from pllm.contracts.semantic_graph import (
    SchedulerTimeline,
    SemanticGraph,
    StateResource,
    StateScope,
    TaskResourceFact,
    TaskResultInputSource,
    TaskSymbol,
    ValueResourceFact,
)
from pllm.contracts.state_ref import LayoutId, LineageId, StateRef, VersionId
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.declaration import Call
from pllm.execution.host_runtime import HostConfig, measured_timeline
from pllm.execution.ptts.request_state import RequestStateSelection, ResourceSelection
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.step_input import TensorStepInput
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import MemoryOperation, TensorBatch

if TYPE_CHECKING:
    from pllm.execution.tensor_binding import TensorBinding


def meta(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A borrowed operand with no storage allocation."""
    return torch.empty(shape, device="meta", dtype=dtype)


def example(
    binding: TensorBinding, facts: BackendCompileFacts, options: ExecutionOptions
) -> tuple[TensorFrame, TensorBatch]:
    """Construct metadata and borrowed operands; execution itself constructs all task scratch."""
    count = facts.max_admitted_lineages if binding.graph.states else facts.probe.max_packed_rows
    rows = tuple(
        PackedPhaseRow(i, i, 0, HiddenPhase.DECODE, RequestIdentity(str(i), i, 0, str(i))) for i in range(count)
    )
    packed = PackedPhaseRowBatch(0, count, rows)
    sampler = PackedSamplerRowBatch(packed, tuple(PackedSamplerRowLink(i, count - i - 1) for i in range(count)))
    # Trace hidden_port's model-dtype gather and FP32 conversion, including their
    # overlapping lifetimes, along with each executor's consumer index table.
    hidden_dtype = facts.probe.tensor_dtype.torch
    hidden = (
        HiddenRowBatch(
            packed,
            tuple(
                HiddenRows(
                    resource, meta(count, facts.probe.hidden_size, dtype=hidden_dtype), np.arange(count, dtype=np.int64)
                )
                for resource in dict.fromkeys(item.binding for item in facts.probe.hidden_resources)
            ),
        )
        if facts.probe.hidden_resources
        else None
    )
    groups: list[ResourceSelection] = []
    roots: list[tuple[torch.Tensor, ...]] = []
    pools: tuple[torch.Tensor, ...] = ()
    resource_pools: dict[str, tuple[torch.Tensor, ...]] = {}
    for state, layout in zip(binding.graph.states, binding.layouts.layouts, strict=True):
        states = count if state.scope is StateScope.REQUEST else 1
        keys = (
            tuple(RequestStateKey.project(row.identity, state.key_fields) for row in rows)
            if state.key_fields
            else (None,)
        )
        refs = StateRefs.of(
            tuple(StateRef(LayoutId(1), LineageId(i + 1 + len(roots)), VersionId(1)) for i in range(states))
        )
        groups.append(
            ResourceSelection(
                state.resource,
                keys,
                refs,
                tuple(len(roots) + i for i in range(count)) if state.key_fields else (len(roots),) * count,
            )
        )
        fields = tuple(meta(states, *field.shape, dtype=field.dtype.torch) for field in layout.layout.fields)
        resource_pools[state.resource] = fields
        if not pools:
            pools = fields
        roots.extend(tuple(field[i] for field in fields) for i in range(states))
    selection = RequestStateSelection(rows, tuple(groups))

    def scoped(resource: StateResource) -> TensorBatch:
        selected = selection.scoped(resource)
        fields = resource_pools[resource]

        def field_slots(field: int) -> tuple[torch.Tensor, tuple[int, ...]]:
            return fields[field], tuple(range(len(fields[field])))

        return replace(
            batch,
            selection=selected,
            roots=tuple(roots[selection.refs.index(ref)] for ref in selected.refs),
            after_refs=selected.refs,
            field_slots=field_slots,
            slot_matrix=lambda: (
                resource,
                np.repeat(np.arange(len(selected.refs), dtype=np.int64)[:, None], len(fields), axis=1),
            ),
        )

    batch = TensorBatch(
        selection,
        tuple(roots),
        selection.refs,
        None,
        select_resource=scoped,
        feedback=SamplerFeedback(
            sampler,
            meta(count, facts.vocab_size or facts.probe.hidden_size),
            meta(count, dtype=torch.long),
        ),
        field_slots=lambda field: (pools[field], tuple(range(len(pools[field])))),
        slot_matrix=lambda: ("", np.repeat(np.arange(len(roots), dtype=np.int64)[:, None], len(pools), axis=1)),
    )
    auxiliary = None
    if binding.graph.training_mode == "weighted_batch":
        keys = tuple(key for key in selection.keys if key is not None)
        shape = (len(keys), options.training_rows) if keys else (options.training_rows,)
        auxiliary = WeightedCEBatch(
            meta(*shape, facts.probe.hidden_size),
            meta(*shape, facts.vocab_size),
            meta(*shape, dtype=torch.long),
            meta(*shape),
            keys,
        )
    dtype = facts.probe.tensor_dtype.torch
    # A step packs at most max_packed_rows prompt tokens across all rows. The first row takes the remainder,
    # so both one longest prompt and every row's fixed cost are traced.
    lengths = (max(1, facts.probe.max_packed_rows - len(rows) + 1), *(1,) * (len(rows) - 1))
    operands = tuple(
        CapturedParameterOperand(
            row.identity,
            target.target,
            HiddenPhase.PREFILL,
            meta(length, target.input_size, dtype=dtype),
            meta(length, target.output_size, dtype=dtype),
            meta(target.output_size, target.input_size, dtype=dtype),
            meta(dtype=torch.bool),
        )
        for row, length in zip(rows, lengths, strict=True)
        for target in facts.probe.targets
    )
    data = TensorFrame(
        TensorStepInput(packed, sampler, hidden, operands=operands, auxiliary=auxiliary),
        rows[::-1],
        replacements=tuple(
            (
                i,
                tuple(
                    torch.empty_like(view.read(root[view.field]))
                    for view in binding.imports[str(selection.addresses[i].resource)]
                ),
            )
            for i, root in enumerate(roots)
            if str(selection.addresses[i].resource) in binding.imports
        ),
    )
    return data, batch


def producers_first(graph: SemanticGraph, symbols: tuple[TaskSymbol, ...]) -> tuple[TaskSymbol, ...]:
    """Keep the declaration order, but move each result producer before its readers.

    Measured operations share one example batch, so a reader of pllm.result finds its producer's
    output only if the producer ran first; operations are declared by name, not by dependency.
    """
    producers = {
        task.symbol: tuple(port.source.task for port in task.inputs if isinstance(port.source, TaskResultInputSource))
        for task in graph.tasks
    }
    ordered: dict[TaskSymbol, None] = {}

    def visit(symbol: TaskSymbol) -> None:
        for producer in producers[symbol]:
            visit(producer)
        ordered[symbol] = None

    for symbol in symbols:
        visit(symbol)
    return tuple(symbol for symbol in ordered if symbol in symbols)


def derive_resources(
    binding: TensorBinding, facts: BackendCompileFacts, options: ExecutionOptions
) -> SchedulerTimeline:
    """Ask the executable operations for their memory; canonical plumbing is funded separately."""
    # Shape-only steps do not allocate a capture pool or consume its eventual entitlement.
    shape = ((facts.max_admitted_lineages, facts.probe.hidden_size),)
    resources = tuple(
        TaskResourceFact(symbol, 0, 1 << 60, shape) for symbol in (*binding.operations, *binding.tensor_tasks)
    )
    data, batch = example(binding, facts, options)
    tasks: list[TaskResourceFact] = []
    values: list[ValueResourceFact] = []
    operations: dict[TaskSymbol, MemoryOperation[TensorFrame]] = {
        **binding.executables(facts, resources),
        **binding.publications,
    }
    for symbol in producers_first(binding.graph, tuple(operations)):
        operation = operations[symbol]
        task = binding.tensor_tasks.get(symbol)
        phase = None if task is None else task.phase
        if phase is not None:
            packed = replace(
                data.step.rows, rows=tuple(replace(row, phase=phase, prompt_length=1) for row in data.step.rows.rows)
            )
            sampler = replace(data.step.sampler_rows, row_batch=packed)
            hidden = None if data.step.hidden is None else replace(data.step.hidden, row_batch=packed)
            data = replace(
                data,
                step=replace(data.step, rows=packed, sampler_rows=sampler, hidden=hidden),
                rows=packed.rows,
                plan=None,
            )
        usage = operation.memory(data, batch)
        shapes = operation.input_shapes(
            len(data.rows),
            facts.probe.hidden_size,
            (
                (len(batch.roots), options.training_rows, facts.probe.hidden_size),
                (len(batch.roots), options.training_rows, facts.vocab_size),
            ),
        )
        tasks.append(
            TaskResourceFact(
                symbol,
                usage.workspace,
                operation.resident(options.capture_memory_bytes),
                shapes,
            )
        )
        values.extend(
            ValueResourceFact(symbol, output.name, usage.output)
            for task in binding.graph.tasks
            if task.symbol == symbol
            for output in task.outputs
        )
    return replace(
        facts.timeline,
        task_resources=tuple(sorted(tasks, key=lambda value: value.task)),
        value_resources=tuple(sorted(values, key=lambda value: (value.task, value.output))),
    )


def derive_host_resources(binding: TensorBinding, facts: HostConfig) -> SchedulerTimeline:
    """Measure a rowless host step over the workload's one instance state at the maximum input rows."""
    if facts.max_input_rows is None:
        raise ValueError("set HostConfig.max_input_rows when memory_budget_bytes is set")
    rows = facts.max_input_rows
    auxiliary = WeightedRowBatch(meta(rows, facts.hidden_size), meta(rows), meta(rows))
    (layout,) = binding.layouts.layouts
    refs = StateRefs.of((StateRef(LayoutId(1), LineageId(1), VersionId(1)),))
    selection = RequestStateSelection((), (ResourceSelection(StateResource(layout.resource), (None,), refs, ()),))
    pools = tuple(meta(1, *field.shape, dtype=field.dtype.torch) for field in layout.layout.fields)
    batch = TensorBatch(
        selection,
        (tuple(pool[0] for pool in pools),),
        refs,
        None,
        field_slots=lambda field: (pools[field], (0,)),
    )
    data = TensorFrame(TensorStepInput.of_batch(auxiliary), ())
    operations: dict[TaskSymbol, MemoryOperation[TensorFrame]] = {
        **binding.executables(None, ()),
        **binding.publications,
    }
    tasks = {
        task.symbol: Call(task.symbol, tuple(output.name for output in task.outputs)) for task in binding.graph.tasks
    }
    costs = [
        (tasks[symbol], operations[symbol].memory(data, batch))
        for symbol in producers_first(binding.graph, tuple(operations))
    ]
    held = sum(
        tensor.numel() * tensor.element_size() for tensor in (auxiliary.hidden, auxiliary.targets, auxiliary.weights)
    )
    return measured_timeline(facts, held, costs, ((rows, facts.hidden_size),))
