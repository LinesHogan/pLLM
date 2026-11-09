"""Memory planning for bound tensor operations.

Builds maximum-shape meta inputs for every operation (example), traces each operation's workspace and capture
residency (derive_resources, derive_host_resources) and orders producers before their readers. Workload
specialization (usage.py) calls it before planning.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch

from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import ModelProbe
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
from pllm.execution.model_sites import EarlierInput, PatchInput, ResultInput, SiteInput, StateInput
from pllm.execution.module_graph import ModuleCall
from pllm.execution.ptts.request_state import RequestStateSelection, ResourceSelection
from pllm.execution.ptts.slot_pools import Slots
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import MemoryOperation, TensorBatch
from pllm.execution.token_program import Segment
from pllm.execution.torch_private import GatherTrace, fixed_shapes

if TYPE_CHECKING:
    from pllm.execution.tensor_declaration import TensorDeclaration


def custom_forward_capture_bytes(binding: TensorDeclaration, probe: ModelProbe) -> int:
    """Estimate native wave graphs from model widths and the workload's traced layer visits.

    Source operators use a no-reuse activation envelope. Each adaptive bucket
    owns library scratch on its conditional stream (64 MiB unless measured larger).
    This is a reservation estimate, not a prediction of allocator residency;
    unusual wave-width/patch combinations can need an explicit capture budget.
    """
    if binding.custom_forward_programs is None:
        return 0
    programs, domain = binding.custom_forward_programs, probe.causal_domain
    assert domain is not None
    rows = domain.requests
    channels = 12 * probe.hidden_size + 6 * probe.intermediate_size
    channels += 4 * (probe.attention_heads + probe.kv_heads) * probe.head_dim
    activation = rows * channels * probe.tensor_dtype.itemsize
    scratch = max(64 << 20, probe.blas_workspace_bytes)
    reservations: list[int] = []
    for program in programs.values():
        if program.over == "span":
            continue
        visits = branches = 0
        for step in program.steps:
            if isinstance(step, Segment):
                visits += step.stop - step.start
                if step.active is not None:
                    branches += rows.bit_length() + 2
                    visits += 2 * (step.stop - step.start)
        reservations.append(2 * activation * visits + scratch * (branches + 1))
    return max(reservations, default=0)


def meta(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A borrowed operand with no storage allocation."""
    return torch.empty(shape, device="meta", dtype=dtype)


def _hidden_storage(facts: BackendCompileFacts, resource: PackedHiddenResourceBinding, count: int) -> torch.Tensor:
    """Shape-only rows of one hidden input: model-dtype hidden rows, or FP32 [layers, vocab] layer logits."""
    if resource.site is HiddenSite.LAYER_LOGITS:
        return meta(count, len(facts.probe.layer_logits), facts.vocab_size, dtype=torch.float32)
    return meta(count, facts.probe.hidden_size, dtype=facts.probe.tensor_dtype.torch)


def _example_field_slots(fields: tuple[torch.Tensor, ...], field: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    return fields[field], tuple(range(len(fields[field])))


def _example_slot_matrix(resource: str, fields: tuple[torch.Tensor, ...]) -> tuple[str, Slots]:
    return resource, np.repeat(np.arange(len(fields[0]), dtype=np.int64)[:, None], len(fields), axis=1)


def example(
    binding: TensorDeclaration, facts: BackendCompileFacts, options: ExecutionOptions, *, tokens: bool = False
) -> tuple[TensorFrame, TensorBatch]:
    """Construct metadata and borrowed operands; execution itself constructs all task scratch.

    `tokens` gives every scheduled token its own row, as readers of tokens="all" see them."""
    # Hidden taps keep one row per request (its prompt end or decode token), as do request states.
    count = (
        facts.max_admitted_lineages
        if (binding.graph.states or facts.probe.hidden_resources) and not tokens
        else facts.probe.max_packed_rows
    )
    rows = tuple(
        PackedPhaseRow(i, i, 0, HiddenPhase.DECODE, RequestIdentity(str(i), i, 0, str(i))) for i in range(count)
    )
    packed = PackedPhaseRowBatch.from_rows(0, count, rows)
    rows = packed.rows
    sampler = PackedSamplerRowBatch.from_links(
        packed, tuple(PackedSamplerRowLink(i, count - i - 1) for i in range(count))
    )
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
                dict(zip(refs.lineages.tolist(), keys, strict=True)),
                refs,
                np.arange(count, dtype=np.int64) + len(roots)
                if state.key_fields
                else np.full(count, len(roots), dtype=np.int64),
            )
        )
        fields = tuple(meta(states, *field.shape, dtype=field.dtype.torch) for field in layout.layout.fields)
        resource_pools[state.resource] = fields
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
        slot_matrix=lambda: ("", np.repeat(np.arange(len(roots), dtype=np.int64)[:, None], len(pools), axis=1)),
    )
    resources: dict[StateResource, TensorBatch] = {}
    for group in selection.groups:
        selected = selection.scoped(group.resource)
        fields = resource_pools[group.resource]
        resources[group.resource] = replace(
            batch,
            selection=selected,
            roots=tuple(roots[int(index)] for index in selection.refs.positions(selected.refs)),
            after_refs=selected.refs,
            field_slots=partial(_example_field_slots, fields),
            slot_matrix=partial(_example_slot_matrix, group.resource, fields),
        )
    batch = replace(batch, resources=resources)
    host_batch = None
    if binding.graph.training_mode == "weighted_batch":
        keys = tuple(key for key in selection.keys if key is not None)
        shape = (len(keys), options.max_host_training_rows) if keys else (options.max_host_training_rows,)
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
    # Operands of a phase the overlay trains on, so every update kind runs (and is funded and checked) here.
    phase = HiddenPhase.PREFILL if binding.overlay_math is None else binding.overlay_math.phases[0]
    operands = tuple(
        ProjectionOperands(
            row.identity,
            target.target,
            phase,
            meta(length, target.input_size, dtype=dtype),
            meta(length, target.target_size, dtype=dtype),
            meta(target.output_size, target.input_size, dtype=dtype),
            meta(dtype=torch.bool),
            meta(length, facts.probe.hidden_size, dtype=dtype),
        )
        for row, length in zip(rows, lengths, strict=True)
        for target in facts.probe.targets
    )
    data = TensorFrame(
        StepInput(
            packed,
            sampler,
            hidden,
            operands=operands,
            batch=host_batch,
            input_tokens=meta(count, binding.graph.input_token_history + 1, dtype=torch.long)
            if binding.graph.input_token_history >= 0
            else None,
        ),
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
    """Select a method's token phase for shared memory and arithmetic shape tracing; example rows are all fresh,
    so every-token readers keep them all."""
    if phase in (None, HiddenPhase.ALL):
        return data
    packed = PackedPhaseRowBatch.from_rows(
        data.step.rows.step_id,
        data.step.rows.total_rows,
        tuple(replace(row, phase=phase, prompt_length=1) for row in data.step.rows.rows),
    )
    sampler = replace(data.step.sampler_rows, row_batch=packed)
    hidden = None if data.step.hidden is None else replace(data.step.hidden, row_batch=packed)
    return replace(
        data, step=replace(data.step, rows=packed, sampler_rows=sampler, hidden=hidden), rows=packed.rows, plan=None
    )


def _site_input_shapes(
    inputs: tuple[tuple[str, SiteInput], ...], batch: TensorBatch
) -> tuple[tuple[str, SiteInput], ...]:
    """Resolve hook buffer shapes from the method outputs already traced for planning."""
    return tuple(
        (
            name,
            replace(
                source,
                shape=tuple(batch.results[source.task, "value"].shape[1:]),
                dtype=batch.results[source.task, "value"].dtype,
            )
            if isinstance(source, ResultInput)
            else source,
        )
        for name, source in inputs
    )


def shape_site_results(binding: TensorDeclaration, facts: BackendCompileFacts) -> None:
    """Trace pre-forward result metadata before backends allocate captured hook buffers."""
    if not binding.model_site_tasks:
        return
    data, batch = example(binding, facts, ExecutionOptions())
    operations = binding.executables(facts, ())
    for symbol in producers_first(binding.graph, tuple(binding.model_site_tasks)):
        with fixed_shapes():
            operations[symbol].memory(data, batch)
    binding.sites.rewrites = [
        replace(
            rewrite,
            inputs=_site_input_shapes(rewrite.inputs, batch),
            read_inputs=_site_input_shapes(rewrite.read_inputs, batch),
        )
        for rewrite in binding.sites.rewrites
    ]


def derive_resources(
    binding: TensorDeclaration, facts: BackendCompileFacts, options: ExecutionOptions
) -> SchedulerTimeline:
    """Ask the executable operations for their memory; framework plumbing is funded separately."""
    # Shape-only steps do not allocate a capture pool or consume its eventual budget.
    resources = tuple(
        TaskResourceFact(
            symbol,
            0,
            1 << 60,
            (
                (
                    facts.probe.max_packed_rows if symbol in binding.model_site_tasks else facts.max_admitted_lineages,
                    facts.probe.hidden_size,
                ),
            ),
        )
        for symbol in (*binding.operations, *binding.tensor_tasks)
    )
    requests = example(binding, facts, options)
    tokens = (
        example(binding, facts, options, tokens=True)
        if any(task.phase is HiddenPhase.ALL for task in binding.tensor_tasks.values())
        else requests
    )
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
        data, batch = tokens if phase is HiddenPhase.ALL else (task_example(requests[0], phase), requests[1])
        tables = tuple(
            (str(address.resource), tensor, state.placement == "host")
            for address, root in zip(batch.selection.addresses, batch.roots, strict=True)
            for state in binding.graph.states
            if state.resource == address.resource and state.immutable
            for tensor in root
        )
        try:
            with fixed_shapes(), GatherTrace(tables) as gathers:
                usage = operation.memory(data, batch)
                gathers.check_outputs(tuple(batch.results.values()))
            for resource, size in gathers.bytes.items():
                binding.gather_bytes[symbol, resource] = size // (
                    facts.probe.max_packed_rows if symbol in binding.model_site_tasks else len(data.rows)
                )
        except (RuntimeError, ValueError, TypeError) as error:
            if "no autograd" in str(error):
                op = str(error).split("through ", 1)[-1].split()[0]
                name = op.split(".")[-2]
                raise ValueError(
                    f"{symbol} trains through {op}, which has no autograd formula; add {name}.register_autograd(...)"
                ) from error
            error.add_note(
                f"while shape tracing method {symbol!r}; methods run inside CUDA graphs: "
                "use tensor operations such as torch.where, not .item() or Python branches on tensor values"
            )
            raise
        if task is not None and task.sampler_stage is not None:
            result = batch.results[symbol, task.outputs[0]]
            additive = task.sampler_stage is SamplerStage.PACKED_PRE_FILTER_ADD
            expected = (len(data.rows), facts.vocab_size if additive else facts.probe.hidden_size)
            if result.shape != expected:
                raise ValueError(
                    f"@pllm.add_logits method `{symbol}` must return [rows, vocab_size] = {expected}; "
                    f"got {tuple(result.shape)}. Read logits=pllm.layer_logits((-1,)) and return logits[:, -1], "
                    "or project hidden states to the model's vocabulary."
                    if additive
                    else f"@pllm.guide_sampling method `{symbol}` must return a hidden-state prediction "
                    f"[rows, hidden_size] = {expected}; got {tuple(result.shape)}. pllm projects it with the model's "
                    "LM head; to add logits directly, use @pllm.add_logits."
                )
        # Model overlays capture bounded tiles, not their whole maximum-shape operation.
        required = operation.resident(usage.capture) if task is not None else 0
        required_capture = max(required_capture, required)
        resident = operation.resident(
            options.cuda_graph_pool_budget_bytes or required or options.model_cuda_graph_pool_budget_bytes
        )
        shapes = operation.input_shapes(
            facts.probe.max_packed_rows if symbol in binding.model_site_tasks else len(data.rows),
            facts.probe.hidden_size,
            (
                (len(batch.roots), options.max_host_training_rows, facts.probe.hidden_size),
                (len(batch.roots), options.max_host_training_rows, facts.vocab_size),
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
    if options.cuda_graph_pool_budget_bytes is not None and options.cuda_graph_pool_budget_bytes < required_capture:
        raise ValueError(
            f"workload needs cuda_graph_pool_budget_bytes >= {required_capture} for its conservative task capture "
            f"reservations; got {options.cuda_graph_pool_budget_bytes}; leave it unset to size from the workload"
        )
    return replace(
        facts.timeline,
        task_resources=tuple(sorted(tasks, key=lambda value: value.task)),
        value_resources=tuple(sorted(values, key=lambda value: (value.task, value.output))),
    )


def derive_host_resources(binding: TensorDeclaration, config: HostConfig) -> SchedulerTimeline:
    """Measure a rowless host step over the workload's one instance state at the maximum input rows."""
    if config.max_input_rows is None:
        raise ValueError("set HostConfig.max_input_rows when execution_peak_budget_bytes is set")
    rows = config.max_input_rows
    host_batch = WeightedRowBatch(meta(rows, config.hidden_size), meta(rows, *config.target_shape), meta(rows))
    (layout,) = binding.layouts.layouts
    refs = StateRefs.of((StateRef(LayoutId(1), LineageId(1), VersionId(1)),))
    selection = RequestStateSelection(
        PackedPhaseRowBatch.from_rows(0, 0, ()).rows,
        (
            ResourceSelection(
                StateResource(layout.resource), {int(refs.lineages[0]): None}, refs, np.empty(0, dtype=np.int64)
            ),
        ),
    )
    pools = tuple(meta(1, *field.shape, dtype=field.dtype.torch) for field in layout.layout.fields)
    batch = TensorBatch(
        selection,
        (tuple(pool[0] for pool in pools),),
        refs,
        None,
        field_slots=lambda field: (pools[field], (0,)),
    )
    data = TensorFrame(StepInput.of_batch(host_batch), PackedPhaseRowBatch.from_rows(0, 0, ()).rows)
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


SiteArgument = torch.Tensor | int | ModuleCall | Callable[[torch.Tensor], torch.Tensor]


def _identity(value: torch.Tensor) -> torch.Tensor:
    return value


def meta_site_state(binding: TensorDeclaration, source: StateInput, rows: int | None) -> torch.Tensor | ModuleCall:
    """A site's state input on meta storage, for planning traces."""
    layout = next(item.layout for item in binding.layouts.layouts if item.resource == source.resource)
    return _site_value(source, tuple(meta(*field.shape, dtype=field.dtype.torch) for field in layout.fields), rows)


def seeded_site_state(
    binding: TensorDeclaration, source: StateInput, rows: int | None, seed: int
) -> torch.Tensor | ModuleCall:
    """A site's state input initialized on CPU from `seed`, for numerical checks."""
    layout = next(item.layout for item in binding.layouts.layouts if item.resource == source.resource)
    root = tuple(torch.zeros(field.shape, dtype=field.dtype.torch) for field in layout.fields)
    binding.initializer(str(source.resource), torch.device("cpu"), seed)((root,), (None,))
    return _site_value(source, root, rows)


def _site_value(source: StateInput, root: tuple[torch.Tensor, ...], rows: int | None) -> torch.Tensor | ModuleCall:
    value = source.view.read(root[source.view.field])
    if source.request and rows is not None:
        value = value.unsqueeze(0).expand(rows, *value.shape)
    return value if source.module is None else ModuleCall(source.module, value, source.request)


def meta_site_arguments(binding: TensorDeclaration, facts: BackendCompileFacts) -> dict[str, dict[str, SiteArgument]]:
    """Residual method examples on meta storage, for cost estimates."""
    rows = facts.max_admitted_lineages
    return _site_arguments(
        binding,
        facts,
        partial(meta_site_state, binding),
        lambda source: meta(rows, *source.shape, dtype=source.dtype),
        torch.device("meta"),
    )


def seeded_site_arguments(
    binding: TensorDeclaration,
    facts: BackendCompileFacts,
    seed: int,
    results: Mapping[tuple[TaskSymbol, str], torch.Tensor],
) -> dict[str, dict[str, SiteArgument]]:
    """Residual method examples with seeded CPU states and the step's task results, for numerical checks."""
    return _site_arguments(
        binding,
        facts,
        partial(seeded_site_state, binding, seed=seed),
        lambda source: results[source.task, "value"],
        torch.device("cpu"),
    )


def _site_arguments(
    binding: TensorDeclaration,
    facts: BackendCompileFacts,
    state: Callable[[StateInput, int | None], torch.Tensor | ModuleCall],
    result: Callable[[ResultInput], torch.Tensor],
    device: torch.device,
) -> dict[str, dict[str, SiteArgument]]:
    arguments: dict[str, dict[str, SiteArgument]] = {}
    for rewrite in binding.sites.rewrites:
        for name, sources in ((rewrite.name, rewrite.inputs), (rewrite.name + ".read", rewrite.read_inputs)):
            kwargs: dict[str, SiteArgument] = {}
            for key, source in sources:
                if isinstance(source, StateInput):
                    rows = (
                        None
                        if rewrite.memory is not None and source == rewrite.memory.carry
                        else facts.max_admitted_lineages
                    )
                    kwargs[key] = state(source, rows)
                elif isinstance(source, ResultInput):
                    kwargs[key] = result(source)
                elif isinstance(source, EarlierInput):
                    kwargs[key] = torch.zeros(
                        (facts.max_admitted_lineages, facts.probe.hidden_size),
                        dtype=facts.probe.tensor_dtype.torch,
                        device=device,
                    )
                elif isinstance(source, PatchInput):
                    kwargs[key] = _identity
                else:
                    kwargs[key] = rewrite.depth(rewrite.places[0])
            arguments[name] = kwargs
    return arguments
