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
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.host_batch import WeightedCEBatch, WeightedRowBatch
from pllm.contracts.packed_hidden import HiddenRowBatch, HiddenRows, PackedHiddenResourceBinding
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.parameter_overlay import ProjectionOperands
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import SamplerFeedback, SamplerStage
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
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import MemoryOperation, TensorBatch

if TYPE_CHECKING:
    from pllm.execution.tensor_binding import TensorBinding


def meta(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A borrowed operand with no storage allocation."""
    return torch.empty(shape, device="meta", dtype=dtype)


def _hidden_storage(facts: BackendCompileFacts, resource: PackedHiddenResourceBinding, count: int) -> torch.Tensor:
    """Shape-only rows of one hidden input: model-dtype hidden rows, or FP32 [layers, vocab] layer logits."""
    if resource.site is HiddenSite.LAYER_LOGITS:
        return meta(count, len(facts.probe.layer_logits), facts.vocab_size, dtype=torch.float32)
    return meta(count, facts.probe.hidden_size, dtype=facts.probe.tensor_dtype.torch)


def example(
    binding: TensorBinding, facts: BackendCompileFacts, options: ExecutionOptions
) -> tuple[TensorFrame, TensorBatch]:
    """Construct metadata and borrowed operands; execution itself constructs all task scratch."""
    # Hidden taps keep one row per request (its prompt end or decode token), as do request states.
    count = (
        facts.max_admitted_lineages
        if binding.graph.states or facts.probe.hidden_resources
        else facts.probe.max_packed_rows
    )
    rows = tuple(
        PackedPhaseRow(i, i, 0, HiddenPhase.DECODE, RequestIdentity(str(i), i, 0, str(i))) for i in range(count)
    )
    packed = PackedPhaseRowBatch(0, count, rows)
    sampler = PackedSamplerRowBatch(packed, tuple(PackedSamplerRowLink(i, count - i - 1) for i in range(count)))
    # Trace hidden_port's model-dtype gather and FP32 conversion, including their
    # overlapping lifetimes, along with each executor's consumer index table.
    hidden = (
        HiddenRowBatch(
            packed,
            tuple(
                HiddenRows(resource, _hidden_storage(facts, resource, count), np.arange(count, dtype=np.int64))
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
    host_batch = None
    if binding.graph.training_mode == "weighted_batch":
        keys = tuple(key for key in selection.keys if key is not None)
        shape = (len(keys), options.max_training_rows) if keys else (options.max_training_rows,)
        batch_dtype = binding.layouts.layouts[0].layout.fields[0].dtype.torch
        host_batch = WeightedCEBatch(
            meta(*shape, facts.probe.hidden_size, dtype=batch_dtype),
            meta(*shape, facts.vocab_size, dtype=batch_dtype),
            meta(*shape, dtype=torch.long),
            meta(*shape, dtype=batch_dtype),
            keys,
        )
    dtype = facts.probe.tensor_dtype.torch
    # A step packs at most max_packed_rows prompt tokens across all rows. The first row takes the remainder,
    # so both one longest prompt and every row's fixed cost are traced.
    lengths = (max(1, facts.probe.max_packed_rows - len(rows) + 1), *(1,) * (len(rows) - 1))
    operands = tuple(
        ProjectionOperands(
            row.identity,
            target.target,
            HiddenPhase.PREFILL,
            meta(length, target.input_size, dtype=dtype),
            meta(length, target.output_size, dtype=dtype),
            meta(target.output_size, target.input_size, dtype=dtype),
            meta(dtype=torch.bool),
            meta(length, facts.probe.hidden_size, dtype=dtype),
        )
        for row, length in zip(rows, lengths, strict=True)
        for target in facts.probe.targets
    )
    data = TensorFrame(
        StepInput(packed, sampler, hidden, operands=operands, batch=host_batch),
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


def task_example(data: TensorFrame, phase: HiddenPhase | None) -> TensorFrame:
    """Select a method's token phase for shared memory and arithmetic shape tracing."""
    if phase is None:
        return data
    packed = replace(
        data.step.rows, rows=tuple(replace(row, phase=phase, prompt_length=1) for row in data.step.rows.rows)
    )
    sampler = replace(data.step.sampler_rows, row_batch=packed)
    hidden = None if data.step.hidden is None else replace(data.step.hidden, row_batch=packed)
    return replace(
        data, step=replace(data.step, rows=packed, sampler_rows=sampler, hidden=hidden), rows=packed.rows, plan=None
    )


def derive_resources(
    binding: TensorBinding, facts: BackendCompileFacts, options: ExecutionOptions
) -> SchedulerTimeline:
    """Ask the executable operations for their memory; framework plumbing is funded separately."""
    # Shape-only steps do not allocate a capture pool or consume its eventual budget.
    shape = ((facts.max_admitted_lineages, facts.probe.hidden_size),)
    resources = tuple(
        TaskResourceFact(symbol, 0, 1 << 60, shape) for symbol in (*binding.operations, *binding.tensor_tasks)
    )
    data, batch = example(binding, facts, options)
    tasks: list[TaskResourceFact] = []
    values: list[ValueResourceFact] = []
    required_capture = 0
    operations: dict[TaskSymbol, MemoryOperation[TensorFrame]] = {
        **binding.executables(facts, resources),
        **binding.publications,
    }
    for symbol in producers_first(binding.graph, tuple(operations)):
        operation = operations[symbol]
        task = binding.tensor_tasks.get(symbol)
        phase = None if task is None else task.phase
        data = task_example(data, phase)
        try:
            usage = operation.memory(data, batch)
        except (RuntimeError, ValueError, TypeError) as error:
            error.add_note(
                f"while shape tracing method {symbol!r}; methods run inside CUDA graphs: "
                "use tensor operations such as torch.where, not .item() or Python branches on tensor values"
            )
            raise
        if task is not None and task.sampler_stage is SamplerStage.PACKED_PRE_FILTER_ADD:
            result = batch.results[symbol, task.outputs[0]]
            expected = (len(data.rows), facts.vocab_size)
            if result.shape != expected:
                raise ValueError(
                    f"@pllm.add_logits method `{symbol}` must return [rows, vocab_size] = {expected}; "
                    f"got {tuple(result.shape)}. Read logits=pllm.layer_logits((-1,)) and return logits[:, -1], "
                    "or project hidden states to the model's vocabulary."
                )
        # Model overlays capture bounded tiles, not their whole maximum-shape operation.
        required = operation.resident(usage.capture) if task is not None else 0
        required_capture = max(required_capture, required)
        resident = operation.resident(options.capture_memory_bytes or required or options.model_capture_bytes)
        shapes = operation.input_shapes(
            len(data.rows),
            facts.probe.hidden_size,
            (
                (len(batch.roots), options.max_training_rows, facts.probe.hidden_size),
                (len(batch.roots), options.max_training_rows, facts.vocab_size),
            ),
        )
        tasks.append(
            TaskResourceFact(
                symbol,
                usage.workspace,
                resident,
                shapes,
            )
        )
        values.extend(
            ValueResourceFact(symbol, output.name, usage.output)
            for task in binding.graph.tasks
            if task.symbol == symbol
            for output in task.outputs
        )
    if options.capture_memory_bytes is not None and options.capture_memory_bytes < required_capture:
        raise ValueError(
            f"workload needs capture_memory_bytes >= {required_capture} for its conservative task capture "
            f"reservations; got {options.capture_memory_bytes}; leave it unset to size from the workload"
        )
    return replace(
        facts.timeline,
        task_resources=tuple(sorted(tasks, key=lambda value: value.task)),
        value_resources=tuple(sorted(values, key=lambda value: (value.task, value.output))),
    )


def derive_host_resources(binding: TensorBinding, config: HostConfig) -> SchedulerTimeline:
    """Measure a rowless host step over the workload's one instance state at the maximum input rows."""
    if config.max_input_rows is None:
        raise ValueError("set HostConfig.max_input_rows when memory_budget_bytes is set")
    rows = config.max_input_rows
    host_batch = WeightedRowBatch(meta(rows, config.hidden_size), meta(rows, *config.target_shape), meta(rows))
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
    data = TensorFrame(StepInput.of_batch(host_batch), ())
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
        tensor.numel() * tensor.element_size() for tensor in (host_batch.hidden, host_batch.targets, host_batch.weights)
    )
    return measured_timeline(config, held, costs, ((rows, config.hidden_size),))
