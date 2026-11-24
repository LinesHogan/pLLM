"""Shape-derived arithmetic and logical traffic estimates used by pllm.explain; never latency predictions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

import torch
from torch.utils.flop_counter import FlopCounterMode

from pllm.config import ExecutionOptions
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite, LayerRouteSelector
from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    StateEffectKind,
    StateScope,
    StateSelectionInputSource,
    TaskSymbol,
)
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.task_memory import example, meta_site_arguments, producers_first, task_example
from pllm.execution.tensor_declaration import TensorDeclaration
from pllm.execution.tensor_program import TensorBatch, TensorOperation
from pllm.execution.token_program import Cell
from pllm.execution.torch_private import (
    CustomOpDef,
    CustomOpTrace,
    FakeTensorMode,
    KernelCountMode,
    custom_op,
    tensor_node_metadata,
)
from pllm.kernels.page_copy import copy_pages


@dataclass(frozen=True)
class OpCost:
    """One opaque operator's declared arithmetic and lower-bound logical traffic per row."""

    name: str
    call_sites: tuple[str, ...]
    flops_per_row: float | None
    bytes_per_row: float
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
        op.cpu,
        op.autograd,
    )


def _estimate_ops(
    binding: TensorDeclaration, facts: BackendCompileFacts, calls: OpCalls
) -> tuple[tuple[OpCost, ...], dict[str, int]]:
    """Add model-site math that task tracing leaves to the native model driver."""
    kwargs = meta_site_arguments(binding, facts)
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

    model_kernels: dict[str, int] = {}
    for layer, method in binding.attention.methods.items():
        probe = facts.probe
        rows = facts.max_admitted_lineages
        context = probe.causal_domain.positions if probe.causal_domain is not None else 1
        q, kv, arguments = attention_example(binding, facts, binding.attention, layer, rows, TraceRead(context))
        with CustomOpTrace() as trace, KernelCountMode() as kernels:
            method.function(q, kv, **arguments)
        if method.name != "native":
            model_kernels[method.name] = model_kernels.get(method.name, 0) + kernels.count
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
                with KernelCountMode() as kernels:
                    cell.graph(*args)
            if role == "decode" and task is not None:
                owner = binding.task_members[task.symbol][0]
                model_kernels[owner] = model_kernels.get(owner, 0) + kernels.count
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
    if binding.overlay_operators:
        layouts = {layout.resource: layout.layout for layout in binding.layouts.layouts}
        target_shapes = {target.target: target for target in facts.probe.targets}
        rows = facts.max_admitted_lineages
        for overlay in binding.effects.overlay_bindings:
            if overlay.operator is not ParameterOverlayOperator.CUSTOM:
                continue
            fn = binding.overlay_operators[overlay.operator_id]
            shape = target_shapes[overlay.target]
            layout = layouts[overlay.resource]
            with FakeTensorMode(), CustomOpTrace() as trace, KernelCountMode() as kernels:
                hidden = torch.empty((rows, shape.input_size), dtype=facts.probe.tensor_dtype.torch, device="meta")
                weights = tuple(
                    torch.empty(view.shape, dtype=layout.fields[view.field].dtype.torch, device="meta")
                    for view in overlay.fields
                )
                fn(hidden, *weights)
            _append_calls(calls, trace, f"Adapter [{overlay.target.name}]", rows)
            adapter_member = next(
                (name for name, decl in binding.members.items() if decl.effect == "Adapter"),
                "Adapter",
            )
            model_kernels[adapter_member] = model_kernels.get(adapter_member, 0) + kernels.count
    return tuple(_op_cost(name, entries) for name, entries in sorted(calls.items())), model_kernels


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """Logical work at the declared decode batch size, ignoring task-body fusion and cache reuse.

    Matmul counts include autograd where declared. Read traffic follows declared fields; transitions count whole roots;
    optimizer traffic is a subset of state traffic, not an additional allocation or DRAM measurement.
    Projection counts cover extra LM-head rows. Shared layer projections and the
    ordered-add sampler copy are charged once; each addition reads its contribution.
    Kernel counts are source operators in the task body; eager counts cover state
    copies and sampler work outside that body. Prefill-only bodies count zero at decode.
    """

    rows: int
    matmul_flops_estimate: int = 0
    projection_flops_estimate: int = 0
    state_read_bytes_estimate: int = 0
    state_write_bytes_estimate: int = 0
    optimizer_bytes_estimate: int = 0
    sampler_bytes_estimate: int = 0
    kernel_count_estimate: int = 0
    eager_kernel_count_estimate: int = 0
    label: str = (
        "estimate: matmuls, logical bytes and source kernels per decode step; ignores cache reuse and task-body fusion"
    )


def _traffic(binding: TensorDeclaration, facts: BackendCompileFacts, symbol: TaskSymbol) -> tuple[int, int]:
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


def _copy_kernels(operation: TensorOperation, batch: TensorBatch) -> int:
    if operation.mode == "read" or operation.copy_fields == ():
        return 0
    batch = operation.scoped(batch)
    with KernelCountMode() as counter:
        for field in range(len(batch.roots[0]) if batch.roots else 0):
            if operation.copy_fields is not None and field not in operation.copy_fields:
                continue
            pool, slots = batch.field_slots(field)
            indices = torch.empty(len(slots), dtype=torch.long, device="meta")
            copy_pages(pool, pool, indices, indices)
    return counter.count


def _sampler_kernels(
    binding: TensorDeclaration,
    facts: BackendCompileFacts,
    symbol: TaskSymbol,
    batch: TensorBatch,
    projected_layers: int,
    padded_vocab_size: int,
    copies_logits: bool,
) -> int:
    from pllm.backends.vllm.guided_sampling import guided_sample
    from pllm.backends.vllm.sampler import GuidedSampler, SignedSamplerGuidance, SignedSamplerStages
    from pllm.contracts.sampler import PackedSamplerGuidanceBatch

    task = binding.tensor_tasks.get(symbol)
    if task is None or (task.sampler_stage is None and not projected_layers):
        return 0
    rows, hidden = facts.max_admitted_lineages, facts.probe.hidden_size
    dtype = facts.probe.tensor_dtype.torch
    logits = torch.empty((rows, facts.vocab_size), dtype=dtype, device="meta")
    prediction = batch.results[symbol, task.outputs[0]] if task.sampler_stage is not None else logits
    ids = torch.empty(rows, dtype=torch.long, device="meta")
    weights = torch.empty((padded_vocab_size, hidden), dtype=dtype, device="meta")
    assert batch.feedback is not None
    guidance = PackedSamplerGuidanceBatch(
        batch.feedback.sampler_rows,
        prediction,
        tuple(range(rows)),
        task.sampler_beta,
        stage=task.sampler_stage or SamplerStage.PACKED_POST_FILTER_EXACT,
    )
    with (
        FakeTensorMode(allow_non_fake_inputs=True),
        KernelCountMode() as counter,
        torch.compiler.set_stance("force_eager"),
    ):
        for _ in range(projected_layers):
            torch.mm(torch.empty((rows, hidden), dtype=dtype, device="meta"), weights.T)
        if task.sampler_beta:
            if facts.guidance_capacity is not None:
                guided_sample(
                    logits,
                    prediction,
                    weights,
                    ids,
                    None,
                    task.sampler_beta,
                    facts.guidance_capacity,
                    ids,
                    logits[:, 0],
                    ids,
                    ids,
                    fp64=False,
                )
            else:
                dense = torch.mm(prediction.to(dtype), weights.T)[:, : facts.vocab_size]
                GuidedSampler.reweight_finite_survivors(logits, SignedSamplerGuidance(guidance, dense))
        elif copies_logits:
            SignedSamplerStages((SignedSamplerGuidance(guidance, prediction),)).add_logits(logits)
    return counter.count


def estimate_tasks(
    binding: TensorDeclaration, facts: BackendCompileFacts, options: ExecutionOptions, *, padded_vocab_size: int
) -> tuple[dict[str, CostEstimate], tuple[OpCost, ...], dict[str, int]]:
    """Trace the production shape-only operation, including training backward and optimizer work."""
    # Decode has one row per request even for stateless layer-logit readers.
    facts = replace(facts, probe=replace(facts.probe, max_packed_rows=facts.max_admitted_lineages))
    data, batch = example(binding, facts, options)
    executables = binding.executables(facts, facts.timeline.task_resources)
    operations = {**executables, **binding.publications}
    estimates: dict[str, CostEstimate] = {}
    calls: OpCalls = {}
    projected: set[int] = set()
    projected_routes: set[LayerRouteSelector] = set()
    first_addition = min(
        (
            task
            for task in binding.tensor_tasks.values()
            if task.sampler_stage is SamplerStage.PACKED_PRE_FILTER_ADD and task.phase is not HiddenPhase.PREFILL
        ),
        key=lambda task: task.sampler_order,
        default=None,
    )
    for symbol in producers_first(binding.graph, tuple(operations)):
        operation = operations[symbol]
        task = binding.tensor_tasks.get(symbol)
        copies_logits = task is not None and task is first_addition
        if (
            binding.graph.trains_on_batch
            and data.step.batch is None
            and task is None
            and symbol not in binding.publications
        ):
            continue  # A missing host batch bypasses its body; zero would misreport untraced model math.
        data = task_example(data, None if task is None else task.phase)
        with FlopCounterMode(display=False) as counter, CustomOpTrace() as trace, KernelCountMode() as kernels:
            operation.memory(data, batch)
        _append_calls(calls, trace, str(symbol), facts.max_admitted_lineages)
        reads, writes = _traffic(binding, facts, symbol)
        projection = sampler = projected_layers = 0
        if task is not None:
            if task.sampler_beta:
                capacity = facts.guidance_capacity
                projected_tokens = padded_vocab_size if capacity is None else min(capacity, facts.vocab_size)
                projection += 2 * facts.max_admitted_lineages * facts.probe.hidden_size * projected_tokens
                width = facts.probe.tensor_dtype.itemsize
                # Bounded traffic covers the scan, overflow selection/tail draw, peaked-row mask,
                # indirect weights and scratch. Dense reads survivors/predictions and writes logits.
                sampler = facts.max_admitted_lineages * (
                    facts.vocab_size * (8 + width)
                    if capacity is None
                    else 16 * facts.vocab_size
                    + projected_tokens * (facts.probe.hidden_size * width + 20)
                    + facts.probe.hidden_size * width
                    + 16
                )
            elif task.sampler_stage is SamplerStage.PACKED_PRE_FILTER_ADD and task.phase is not HiddenPhase.PREFILL:
                width = facts.probe.tensor_dtype.itemsize
                delta = batch.results[symbol, task.outputs[0]]
                # One shared sampler-input copy; later declarations only read their contribution.
                sampler = (
                    facts.max_admitted_lineages
                    * facts.vocab_size
                    * (delta.element_size() + (width + 4 if copies_logits else 0))
                )
            if any(
                isinstance(source, HiddenInputSource)
                and source.site is HiddenSite.LAYER_LOGITS
                and source.route is None
                for source in task.sources.values()
            ):
                layers = set(facts.probe.layer_logits) - projected
                projected_layers += len(layers)
                projection += (
                    len(layers) * 2 * facts.max_admitted_lineages * facts.probe.hidden_size * padded_vocab_size
                )
                projected.update(layers)
            routed_logit_sources = {
                source.route
                for source in task.sources.values()
                if isinstance(source, HiddenInputSource)
                and source.site is HiddenSite.LAYER_LOGITS
                and source.route is not None
            } - projected_routes
            if routed_logit_sources:
                projected_layers += len(routed_logit_sources)
                projection += (
                    len(routed_logit_sources)
                    * 2
                    * facts.max_admitted_lineages
                    * facts.probe.hidden_size
                    * padded_vocab_size
                )
                projected_routes.update(routed_logit_sources)
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
            0 if task is not None and task.phase is HiddenPhase.PREFILL else kernels.count,
            0
            if task is not None and task.phase is HiddenPhase.PREFILL
            else (
                (0 if symbol not in executables else _copy_kernels(executables[symbol], batch))
                + _sampler_kernels(binding, facts, symbol, batch, projected_layers, padded_vocab_size, copies_logits)
            ),
        )
    ops, model_kernels = _estimate_ops(binding, facts, calls)
    return estimates, ops, model_kernels
