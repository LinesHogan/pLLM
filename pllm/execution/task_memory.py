"""Maximum-shape invocation inputs used to trace the bound tensor operations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from pllm.authoring import BackendCompileFacts, TensorStepInput
from pllm.config import ExecutionOptions
from pllm.contracts.auxiliary import WeightedCEBatch
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_hidden import PackedHiddenSnapshot, PackedHiddenSnapshotBatch
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.parameter_overlay import CapturedParameterOperand
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.sampler import SamplerFeedback
from pllm.contracts.semantic_graph import SchedulerTimeline, StateScope, TaskResourceFact, TaskSymbol, ValueResourceFact
from pllm.contracts.state_ref import LayoutId, LineageId, StateRef, VersionId
from pllm.execution.ptts.request_state import RequestStateSelection, ResourceSelection
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.shared_tensor import tensor_dtype
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
    packed = PackedPhaseRowBatch(1, 0, count, rows)
    sampler = PackedSamplerRowBatch(packed, tuple(PackedSamplerRowLink(i, count - i - 1) for i in range(count)))
    hidden_dtype = torch.float64 if tensor_dtype(facts.probe.tensor_dtype) is torch.float64 else torch.float32
    hidden = (
        PackedHiddenSnapshotBatch(
            packed,
            tuple(
                PackedHiddenSnapshot(resource.binding, meta(count, facts.probe.hidden_size, dtype=hidden_dtype))
                for resource in facts.probe.hidden_resources
            ),
        )
        if facts.probe.hidden_resources
        else None
    )
    groups: list[ResourceSelection] = []
    roots: list[tuple[torch.Tensor, ...]] = []
    pools: tuple[torch.Tensor, ...] = ()
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
            ResourceSelection(state.resource, keys, refs, tuple(range(count)) if state.key_fields else (0,) * count)
        )
        fields = tuple(meta(states, *field.shape, dtype=tensor_dtype(field.dtype)) for field in layout.layout.fields)
        if not pools:
            pools = fields
        roots.extend(tuple(field[i] for field in fields) for i in range(states))
    selection = RequestStateSelection(rows, tuple(groups))
    batch = TensorBatch(
        selection,
        tuple(roots),
        selection.refs,
        None,
        feedback=SamplerFeedback(
            sampler,
            meta(count, facts.vocab_size or facts.probe.hidden_size),
            meta(count, dtype=torch.long),
        ),
        field_slots=lambda field: (pools[field], tuple(range(len(pools[field])))),
    )
    auxiliary = None
    if binding.weighted_training:
        keys = tuple(key for key in selection.keys if key is not None)
        shape = (len(keys), options.training_rows) if keys else (options.training_rows,)
        auxiliary = WeightedCEBatch(
            meta(*shape, facts.probe.hidden_size),
            meta(*shape, facts.vocab_size),
            meta(*shape, dtype=torch.long),
            meta(*shape),
            keys,
        )
    dtype = tensor_dtype(facts.probe.tensor_dtype)
    operands = tuple(
        CapturedParameterOperand(
            row.identity,
            target.target,
            HiddenPhase.PREFILL,
            meta(facts.probe.max_packed_rows, target.input_size, dtype=dtype),
            meta(facts.probe.max_packed_rows, target.output_size, dtype=dtype),
            meta(target.output_size, target.input_size, dtype=dtype),
        )
        for row in rows
        for target in facts.targets
    )
    data = TensorFrame(
        TensorStepInput(packed, sampler, hidden, operands=operands, auxiliary=auxiliary),
        rows[::-1],
        replacements=tuple((i, tuple(torch.empty_like(value) for value in root)) for i, root in enumerate(roots)),
    )
    return data, batch


def derive_resources(
    binding: TensorBinding, facts: BackendCompileFacts, options: ExecutionOptions
) -> SchedulerTimeline:
    """Ask the executable operations for their memory; canonical plumbing is funded separately."""
    # Shape-only steps do not allocate a capture pool or consume its eventual entitlement.
    shape = ((facts.max_admitted_lineages, facts.probe.hidden_size),)
    binding.resources = tuple(TaskResourceFact(symbol, 0, 1 << 60, shape) for symbol in binding.tensor_tasks)
    binding.configure_steps(facts, options.capture)
    data, batch = example(binding, facts, options)
    tasks: list[TaskResourceFact] = []
    values: list[ValueResourceFact] = []
    operations: dict[TaskSymbol, MemoryOperation[TensorFrame]] = {**binding.operations, **binding.publications}
    for symbol, operation in operations.items():
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
                operation.resident(options.capture_memory_bytes if options.capture else 0),
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
