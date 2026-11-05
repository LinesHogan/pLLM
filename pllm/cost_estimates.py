"""Shape-derived arithmetic and logical traffic estimates used by pllm.explain; never latency predictions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

import torch
from torch.utils.flop_counter import FlopCounterMode

from pllm.config import ExecutionOptions
from pllm.contracts.hidden_locator import HiddenSite
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    StateEffectKind,
    StateScope,
    StateSelectionInputSource,
    TaskSymbol,
)
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.freeze import fingerprint
from pllm.execution.task_memory import example, producers_first, site_arguments, task_example
from pllm.execution.tensor_binding import TensorBinding
from pllm.execution.token_program import Cell
from pllm.execution.torch_private import CustomOpDef, CustomOpTrace, FakeTensorMode, custom_op, tensor_node_metadata


@dataclass(frozen=True)
class OpCost:
    """One opaque operator's declared arithmetic and lower-bound logical traffic per row."""

    name: str
    call_sites: tuple[str, ...]
    flops_per_row: float | None
    bytes_per_row: float
    signature: str
    cpu_reference: bool
    autograd: bool


# Entries retain each call site; role programs are alternatives, so the per-call bound is their maximum.
OpCalls = dict[str, list[tuple[str, float | None, float, CustomOpDef]]]


def _append_calls(calls: OpCalls, trace: CustomOpTrace, site: str, rows: int) -> None:
    for definition, flops, size in trace.calls:
        op = custom_op(definition)
        assert op is not None
        calls.setdefault(op.name, []).append((site, None if flops is None else flops / rows, size / rows, definition))


def _op_cost(name: str, entries: list[tuple[str, float | None, float, CustomOpDef]]) -> OpCost:
    definition = entries[0][3]
    op = custom_op(definition)
    assert op is not None
    return OpCost(
        name,
        tuple(dict.fromkeys(entry[0] for entry in entries)),
        None if any(entry[1] is None for entry in entries) else max(entry[1] or 0 for entry in entries),
        max(entry[2] for entry in entries),
        fingerprint(definition),
        op.cpu,
        op.autograd,
    )


def _estimate_ops(binding: TensorBinding, facts: BackendCompileFacts, calls: OpCalls) -> tuple[OpCost, ...]:
    """Add model-site math that task tracing leaves to the native model driver."""
    kwargs = site_arguments(binding, facts)
    for rewrite in binding.sites.rewrites:
        rows = facts.max_admitted_lineages if rewrite.memory is None else rewrite.memory.chunk
        x = torch.empty((rows, facts.probe.hidden_size), dtype=facts.probe.tensor_dtype.torch, device="meta")
        stream = x if rewrite.streams == 1 else x.unsqueeze(1).expand(-1, rewrite.streams, -1)
        for name, function, args in (
            (rewrite.name, rewrite.write, (stream,) if rewrite.site in ("value", "final") else (stream, x)),
            (rewrite.name + ".read", rewrite.read, (stream,)),
        ):
            if function is not None:
                with CustomOpTrace() as trace:
                    function(*args, **kwargs[name])
                _append_calls(calls, trace, name, rows)
    from pllm.execution.attention_methods import TraceRead, attention_example

    for layer, method in binding.attention.methods.items():
        probe = facts.probe
        rows = facts.max_admitted_lineages
        context = probe.causal_domain.positions if probe.causal_domain is not None else 1
        q, kv, arguments = attention_example(binding, facts, binding.attention, layer, rows, TraceRead(context))
        with CustomOpTrace() as trace:
            method.function(q, kv, **arguments)
        _append_calls(calls, trace, f"{method.name} [attention layer {layer}]", rows)
    task = binding.graph.custom_forward_task
    for role, program in (binding.custom_forward_programs or {}).items():
        for cell in (step for step in program.steps if isinstance(step, Cell)):
            with FakeTensorMode(), CustomOpTrace() as trace:
                args = tuple(
                    torch.empty(tuple(int(n) for n in value.shape), dtype=value.dtype, device=value.device)
                    for node in cell.graph.graph.nodes
                    if node.op == "placeholder"
                    for value in (tensor_node_metadata(node),)
                    if value is not None
                )
                cell.graph(*args)
            invariants = {name for name, _ in program.parameters}
            rows = next(
                (
                    int(value.shape[0])
                    for name, value in zip(cell.inputs, args, strict=True)
                    if value.ndim and name not in invariants
                ),
                1,
            )
            _append_calls(calls, trace, f"{task.symbol if task else 'forward'} [{role}]", rows)
    return tuple(_op_cost(name, entries) for name, entries in sorted(calls.items()))


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Logical work at the declared decode batch size, ignoring fusion and cache reuse.

    Matmul counts include autograd where declared. Read traffic follows declared fields; transitions count whole roots;
    optimizer traffic is a subset of state traffic, not an additional allocation or DRAM measurement.
    Projection counts cover extra LM-head rows. Shared layer projections are charged once.
    """

    rows: int
    matmul_flops_estimate: int = 0
    projection_flops_estimate: int = 0
    state_read_bytes_estimate: int = 0
    state_write_bytes_estimate: int = 0
    optimizer_bytes_estimate: int = 0
    sampler_bytes_estimate: int = 0
    label: str = "estimate: matmuls and logical bytes per decode step; ignores cache reuse and fusion"


def _traffic(binding: TensorBinding, facts: BackendCompileFacts, symbol: TaskSymbol) -> tuple[int, int]:
    task = next(task for task in binding.graph.tasks if task.symbol == symbol)
    scopes = {str(state.resource): state.scope for state in binding.graph.states}
    sizes = {
        layout.resource: sum(prod(field.shape) * field.dtype.itemsize for field in layout.layout.fields)
        * (facts.max_admitted_lineages if scopes[layout.resource] is StateScope.REQUEST else 1)
        for layout in binding.layouts.layouts
    }
    tensor_task = binding.tensor_tasks.get(symbol)
    reads = writes = 0
    layouts = {layout.resource: layout.layout for layout in binding.layouts.layouts}
    for effect in task.effects:
        if effect.kind is StateEffectKind.PUBLISH:
            continue  # Publication changes ownership metadata; the preceding transition moved the tensors.
        if effect.kind is not StateEffectKind.READ:
            reads += sizes[effect.resource]
            writes += sizes[effect.resource]
        elif tensor_task is None:
            reads += sizes[effect.resource]
        else:
            fields = {
                port.field
                for port in tensor_task.ports.values()
                if isinstance(port.source, StateSelectionInputSource)
                and port.source.resource == effect.resource
                and port.field is not None
            }
            roots = facts.max_admitted_lineages if scopes[effect.resource] is StateScope.REQUEST else 1
            reads += roots * sum(
                prod(field.shape) * field.dtype.itemsize
                for field in layouts[effect.resource].fields
                if field.field_id in fields
            )
    return reads, writes


def estimate_tasks(
    binding: TensorBinding, facts: BackendCompileFacts, options: ExecutionOptions, *, padded_vocab_size: int
) -> tuple[dict[str, CostEstimate], tuple[OpCost, ...]]:
    """Trace the production shape-only operation, including training backward and optimizer work."""
    # Decode has one row per request even for stateless layer-logit readers.
    facts = replace(facts, probe=replace(facts.probe, max_packed_rows=facts.max_admitted_lineages))
    data, batch = example(binding, facts, options)
    executables = binding.executables(facts, facts.timeline.task_resources)
    operations = {**executables, **binding.publications}
    estimates: dict[str, CostEstimate] = {}
    calls: OpCalls = {}
    projected: set[int] = set()
    for symbol in producers_first(binding.graph, tuple(operations)):
        operation = operations[symbol]
        task = binding.tensor_tasks.get(symbol)
        if (
            binding.graph.trains_on_batch
            and data.step.batch is None
            and task is None
            and symbol not in binding.publications
        ):
            continue  # A missing host batch bypasses its body; zero would misreport untraced model math.
        data = task_example(data, None if task is None else task.phase)
        with FlopCounterMode(display=False) as counter, CustomOpTrace() as trace:
            operation.memory(data, batch)
        _append_calls(calls, trace, str(symbol), facts.max_admitted_lineages)
        reads, writes = _traffic(binding, facts, symbol)
        projection = sampler = 0
        if task is not None:
            if task.sampler_beta:
                capacity = options.guidance_capacity
                projected_tokens = padded_vocab_size if capacity is None else min(capacity, facts.vocab_size)
                projection += 2 * facts.max_admitted_lineages * facts.probe.hidden_size * projected_tokens
                width = facts.probe.tensor_dtype.itemsize
                # Bounded figures are capacity upper bounds, including indirect weight reads and
                # scratch. Dense reweight reads FP32 survivors and native predictions, writes FP32.
                sampler = facts.max_admitted_lineages * (
                    facts.vocab_size * (8 + width)
                    if capacity is None
                    else 4 * facts.vocab_size
                    + projected_tokens * (facts.probe.hidden_size * width + 20)
                    + facts.probe.hidden_size * width
                    + 16
                )
            elif task.sampler_stage is SamplerStage.PACKED_PRE_FILTER_ADD:
                width = facts.probe.tensor_dtype.itemsize
                delta = batch.results[symbol, task.outputs[0]]
                conversion = 0 if delta.dtype == facts.probe.tensor_dtype.torch else delta.element_size() + width
                sampler = facts.max_admitted_lineages * facts.vocab_size * (3 * width + conversion)
            if any(
                isinstance(source, HiddenInputSource) and source.site is HiddenSite.LAYER_LOGITS
                for source in task.sources.values()
            ):
                layers = set(facts.probe.layer_logits) - projected
                projection += (
                    len(layers) * 2 * facts.max_admitted_lineages * facts.probe.hidden_size * padded_vocab_size
                )
                projected.update(layers)
        estimates[symbol] = CostEstimate(
            facts.max_admitted_lineages,
            sum(
                counter.get_flop_counts().get("Global", {}).get(operation, 0)
                for operation in (torch.ops.aten.mm, torch.ops.aten.bmm, torch.ops.aten.addmm, torch.ops.aten.baddbmm)
            ),
            projection,
            reads,
            writes,
            reads + writes
            if (symbol in executables and executables[symbol].optimizer is not None)
            or any(binding.members[name].effect == "train" for name in binding.task_members.get(symbol, ()))
            else 0,
            sampler,
        )
    return estimates, _estimate_ops(binding, facts, calls)
