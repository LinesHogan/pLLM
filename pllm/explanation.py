"""CPU planning reports for workload authors, exposed as pllm.explain and JSON-ready dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import TYPE_CHECKING

from pllm.backends.vllm.kv_layout import layout_capacity
from pllm.config import ExecutionOptions, ModelConfig
from pllm.contracts.backend_attachment import TOKEN_CAPTURE_CAPABILITY, AttachmentGraphMode, BackendAttachment
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from pllm.contracts.host_batch import CausalTrainBatch
from pllm.contracts.kv_layout import KVCapacity
from pllm.contracts.member import MemberDeclaration
from pllm.contracts.semantic_graph import (
    HiddenInputSource,
    HostServiceInputSource,
    SemanticTask,
    StateEffectKind,
    StateSelectionInputSource,
    TaskResourceFact,
)
from pllm.contracts.tensor_layout import TensorFieldSpec
from pllm.cost_estimates import CostEstimate, OpCost, estimate_tasks
from pllm.execution.attention_methods import AttentionCost
from pllm.execution.preemption import preemption_policy
from pllm.execution.ptts.page_bundle_planning import page_stride_bytes
from pllm.execution.trace import EVENT_CAPACITY, MAX_SNAPSHOT_BYTES

if TYPE_CHECKING:
    from pllm.backends.vllm.hidden_taps import HiddenTapLayout
    from pllm.backends.vllm.model_shape import ModelShape
    from pllm.execution.compile_facts import BackendCompileFacts
    from pllm.execution.tensor_declaration import TensorDeclaration
    from pllm.workload import Workload


@dataclass(frozen=True, slots=True)
class TaskCost:
    """One planned task; byte counts and version distances come directly from the planner."""

    name: str
    anchor: str
    lane: str
    captured: bool
    reads: tuple[str, ...]
    read_origin: str
    read_distance: int
    workspace_bytes: int
    capture_resident_bytes: int
    estimate: CostEstimate | None


@dataclass(frozen=True, slots=True)
class KernelEstimate:
    """Added source operators per decode step, separated by their launch context."""

    captured: int = 0
    eager: int = 0

    @property
    def total(self) -> int:
        """Operators within graphs plus individually launched eager operators."""
        return self.captured + self.eager

    def __str__(self) -> str:
        context = "graph + eager" if self.captured and self.eager else "graph" if self.captured else "eager"
        return f"~{self.total} ({context})" if self.total else "0"


@dataclass(frozen=True, slots=True)
class MemberCost:
    """An authored method or model value and all tasks emitted on its behalf."""

    name: str
    effect: str
    tasks: tuple[TaskCost, ...]
    eager_reason: str | None
    host_batch: str
    training: str
    kernels: KernelEstimate


@dataclass(frozen=True, slots=True)
class StateCost:
    """Exact physical capacity, including allocator alignment outside each root."""

    name: str
    scope: str
    # Pages per field; older versions keep only the fields their readers pin, so optimizer fields need fewer.
    pages: tuple[int, ...]
    bytes_per_root: int
    reserved_bytes: int
    max_admitted_lineages: int
    fields: tuple[TensorFieldSpec, ...]
    source: str
    placement: str
    host_bytes: int
    transfer_bytes_per_row: int
    transfer_bytes_per_step: int


@dataclass(frozen=True, slots=True)
class EscapeHatch:
    """A timing or execution setting and its consequence; defaults are identified explicitly."""

    member: str
    setting: str
    value: str
    is_default: bool
    consequence: str


@dataclass(frozen=True, slots=True)
class AnchorCost:
    """Persistent input and attachment reservations at one engine anchor."""

    name: str
    reserved_bytes: int


@dataclass(frozen=True, slots=True)
class TapCost:
    """One shared tap allocation and the authored consumers that use it."""

    module: str
    output: int
    phases: tuple[str, ...]
    resident_bytes: int
    members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExportCost:
    """One export's device-to-host copy per step, against the trace's bounded event buffer.

    `rows` is what the export keeps of the `source_rows` a step can carry (max_num_batched_tokens for
    tokens="all", else max_num_seqs); bytes and events assume 8-byte values, an upper bound.
    """

    name: str
    rows: int
    source_rows: int
    columns: int
    bytes: int
    events: int
    overflow: str


@dataclass(frozen=True, slots=True)
class ExplainResult:
    """Exact reservations and labelled arithmetic estimates; asdict is JSON-ready, with no latency prediction."""

    model: ModelShape
    dtype: str
    tensor_parallel_size: int
    eager_reason: str
    members: tuple[MemberCost, ...]
    framework_tasks: tuple[TaskCost, ...]
    states: tuple[StateCost, ...]
    anchors: tuple[AnchorCost, ...]
    taps: tuple[TapCost, ...]
    peak_memory_bytes: int
    memory_label: str
    overlap_threshold_bytes: int | None
    inline_peak_memory_bytes: int
    budget_errors: tuple[str, ...]
    preemption: str
    recompute_rejection: str | None
    workload_host_syncs: int | None
    exports: tuple[ExportCost, ...]
    escape_hatches: tuple[EscapeHatch, ...]
    limitations: tuple[str, ...]
    custom_ops: tuple[OpCost, ...]
    attention: tuple[AttentionCost, ...]
    kv_capacity: tuple[KVCapacity, ...]
    kernels: KernelEstimate
    # Base forward plus every task's matmul and projection FLOPs per row; None when the base is unknown.
    flops_per_token_estimate: float | None
    shared_layer_groups: tuple[tuple[int, ...], ...] = ()

    def __str__(self) -> str:
        forward = f"eager: {self.eager_reason}" if self.eager_reason else "CUDA graphs: no effect needs eager"
        lines = [
            f"model    {self.model.architecture}, hidden {self.model.hidden_size}, {self.model.layer_count} layers, "
            f"vocab {self.model.vocab_size}, {self.dtype} [exact, config.json]",
            f"parallel tensor_parallel_size={self.tensor_parallel_size}; reservations are per rank",
            f"forward  {forward}",
            f"base     {self.model.base_forward_flops_per_token_estimate} FLOP/token [estimate: dense projections and LM head; excludes attention over KV history]",
            "member  effect  anchor / lane / graph  state reads (origin, distance)",
        ]
        for member in self.members:
            lines.append(f"kernels  {member.name}: {member.kernels} / step")
            for task in member.tasks:
                lines.append(
                    f"{member.name}  {member.effect}  {task.anchor} / {task.lane} / "
                    f"{'captured' if task.captured else 'eager'}  "
                    f"{', '.join(task.reads) or '-'} ({task.read_origin}, {task.read_distance})"
                    + (f"; {member.training}" if member.training else "")
                )
                if task.capture_resident_bytes:
                    lines.append(
                        f"  CUDA-graph pool reservation: {task.capture_resident_bytes} bytes [planned capture residency; cuda_graph_pool_budget_bytes caps the pool]"
                    )
                if task.estimate is not None:
                    estimate = task.estimate
                    lines.append(
                        f"  estimate / step ({estimate.rows} rows): {estimate.matmul_flops_estimate} matmul FLOP + "
                        f"{estimate.projection_flops_estimate} projection FLOP; "
                        f"state read/write {estimate.state_read_bytes_estimate}/{estimate.state_write_bytes_estimate} bytes, "
                        f"optimizer {estimate.optimizer_bytes_estimate} bytes (included), sampler {estimate.sampler_bytes_estimate} bytes"
                    )
                else:
                    lines.append(
                        "  matmul/byte estimate / step: unavailable; profile this model-site or host-batch body with a loaded model"
                    )
            if not member.tasks:
                lines.append(f"{member.name}  {member.effect}  model site or inactive method")
            if member.host_batch:
                lines.append(f"  {member.host_batch}")
            if member.eager_reason:
                lines.append(f"  forward eager because {member.name}: {member.eager_reason}")
        lines.append(f"kernels  total: {self.kernels} / step [estimate: source operators; excludes base model]")
        lines.extend(
            f"tap      {tap.module}[{tap.output}], {','.join(tap.phases)}: "
            f"{tap.resident_bytes} bytes shared by {', '.join(tap.members)}"
            for tap in self.taps
        )
        lines.append("state  scope  bytes/root  reserved bytes [exact; state_page_budget_bytes]")
        lines.extend(
            f"{state.name}  {state.scope}  {state.bytes_per_root}  {state.reserved_bytes}  "
            + ", ".join(
                f"field {field.field_id}: {field.dtype} x {pages} pages"
                for field, pages in zip(state.fields, state.pages, strict=True)
            )
            + f"; source: {state.source}; placement: {state.placement}; host bytes: {state.host_bytes}; "
            + f"logical transfer bytes/row: {state.transfer_bytes_per_row}; decode bytes/step: {state.transfer_bytes_per_step}"
            for state in self.states
        )
        lines.append(
            f"memory   peak {self.peak_memory_bytes} bytes funded by pLLM [{self.memory_label}; execution_peak_budget_bytes]"
        )
        if self.overlap_threshold_bytes is not None:
            lines.append(
                f"         below {self.overlap_threshold_bytes} bytes tasks run inline; "
                f"below {self.inline_peak_memory_bytes} bytes planning fails"
            )
        lines.extend(f"budget   {error}" for error in self.budget_errors)
        lines.append(
            f"preempt  state {self.preemption}; replay rejection: {self.recompute_rejection if self.preemption == 'reset' else 'not applicable to kept state'}"
        )
        lines.append(
            f"host     {self.workload_host_syncs if self.workload_host_syncs is not None else 'unverified'} syncs in traced workload math"
        )
        lines.extend(
            f"export   {item.name}: {item.rows} of up to {item.source_rows} rows x {item.columns} columns / step, "
            f"<= {item.bytes} bytes device-to-host in {item.events} trace events [8-byte values; the trace buffers "
            f"{EVENT_CAPACITY} events of <= {MAX_SNAPSHOT_BYTES} bytes]; overflow={item.overflow}"
            + ("; rows beyond max_rows are dropped" if item.rows < item.source_rows else "")
            for item in self.exports
        )
        lines.extend(
            f"timing   {item.member + ': ' if item.member else ''}{item.setting}={item.value} "
            f"({'default' if item.is_default else 'you set'}): {item.consequence}"
            for item in self.escape_hatches
        )
        if self.model.read_only_reason:
            lines.append(f"model    read-only: {self.model.read_only_reason}")
        if self.shared_layer_groups:
            tied = sum(len(group) for group in self.shared_layer_groups)
            lines.append(
                f"shared   {len(self.shared_layer_groups)} groups ({tied} -> {len(self.shared_layer_groups)} unique linear layers)"
            )
        lines.extend(f"note     {note}" for note in self.limitations)
        for op in self.custom_ops:
            flops = (
                f"{op.flops_per_row:g} [declared]"
                if op.flops_per_row is not None
                else "unknown [no flop formula, counted 0]"
            )
            lines.append(
                f"op       {op.name}: {', '.join(op.call_sites)}; {flops} FLOP/row; "
                f"{op.bytes_per_row:g} logical bytes/row [lower bound]; "
                f"CPU reference={op.cpu_reference}, autograd={op.autograd}; host syncs unverified (fake only)"
            )
        for read in self.attention:
            lines.append(
                f"attention layer {read.layer}: custom `{read.method}`; {', '.join(read.reads) or 'user kernels'}; "
                f"{read.flops_per_token} FLOP/token, <= {read.kv_bytes_per_token} KV bytes/token [estimate at max_model_len; per-head budget upper bound]"
            )
        for capacity in self.kv_capacity:
            lines.append(
                f"KV layer {capacity.layout.layer}: {capacity.layout.description}; "
                f"<= {capacity.blocks} blocks, {capacity.bytes} KV bytes/request "
                f"[native admission bound, block size {capacity.block_size}, includes in-flight tokens; "
                "prompt sink worst case]; prefix caching disabled"
            )
        return "\n".join(lines)


def _exports(binding: TensorDeclaration, facts: BackendCompileFacts) -> tuple[ExportCost, ...]:
    result: list[ExportCost] = []
    for task in binding.tensor_tasks.values():
        if task.export is None:
            continue
        export = task.export
        source = facts.probe.max_packed_rows if task.phase is HiddenPhase.ALL else facts.max_admitted_lineages
        rows = min(export.max_rows, source)
        events = ceil(rows / (MAX_SNAPSHOT_BYTES // (export.max_cols * 8)))
        result.append(
            ExportCost(export.name, rows, source, export.max_cols, rows * export.max_cols * 8, events, export.overflow)
        )
    return tuple(result)


def _task_costs(plan: PlanBundle, estimates: dict[str, CostEstimate]) -> dict[str, TaskCost]:
    resources = {item.task: item for item in plan.timeline.task_resources}
    costs: dict[str, TaskCost] = {}
    for physical in plan.schedule.tasks:
        task = plan.semantic_graph.tasks[physical.task_id]
        resource = resources.get(task.symbol, TaskResourceFact(task.symbol, 0))
        implementation = next(
            item for item in task.implementations if item.implementation_id == physical.implementation_id
        )
        costs[task.symbol] = TaskCost(
            task.symbol,
            plan.timeline.anchors[physical.anchor_id].symbol,
            physical.lane,
            any(
                cap in implementation.required_capabilities
                for cap in ("compute/cuda_graph@1", TOKEN_CAPTURE_CAPABILITY)
            ),
            tuple(
                dict.fromkeys(
                    (
                        *(
                            port.source.resource
                            for port in task.inputs
                            if isinstance(port.source, StateSelectionInputSource)
                        ),
                        *(effect.resource for effect in task.effects if effect.kind is StateEffectKind.READ),
                    )
                )
            ),
            physical.read_origin,
            physical.read_distance,
            resource.workspace_bytes,
            resource.resident_bytes,
            estimates.get(task.symbol),
        )
    return costs


def _members(
    binding: TensorDeclaration,
    costs: dict[str, TaskCost],
    reasons: dict[str, str],
    max_host_training_rows: int,
    sites: dict[str, KernelEstimate],
) -> tuple[MemberCost, ...]:
    result: list[MemberCost] = []
    for name, member in binding.members.items():
        tasks = tuple(costs[symbol] for symbol, names in binding.task_members.items() if name in names)
        causal = any(
            name in binding.task_members.get(task.symbol, ())
            and any(
                isinstance(port.source, HostServiceInputSource) and port.source.schema == CausalTrainBatch.schema
                for port in task.inputs
            )
            for task in binding.graph.tasks
        )
        host_batch = (
            f"host batch CausalTrainBatch ({', '.join(CausalTrainBatch.names)}), at most {max_host_training_rows} (max_host_training_rows) "
            "sequences, one AdamW step at the start of generate(batch=...)"
            if causal
            else ""
        )
        training = (
            "one AdamW step per state per engine step; prompt rows train once per prefill chunk"
            if any(
                task.training is not None and name in binding.task_members.get(task.symbol, ())
                for task in binding.tensor_tasks.values()
            )
            else ""
        )
        kernels = _kernel_sum(tasks, (sites.get(name, KernelEstimate()),))
        result.append(MemberCost(name, member.effect, tasks, reasons.get(member.effect), host_batch, training, kernels))
    return tuple(result)


def _flops_per_token(base: int | None, tasks: tuple[TaskCost, ...]) -> float | None:
    if base is None:
        return None
    return base + sum(
        (task.estimate.matmul_flops_estimate + task.estimate.projection_flops_estimate) / task.estimate.rows
        for task in tasks
        if task.estimate is not None
    )


def _kernel_sum(tasks: tuple[TaskCost, ...], sites: tuple[KernelEstimate, ...]) -> KernelEstimate:
    graph = sum(site.captured for site in sites)
    eager = sum(site.eager for site in sites)
    for task in tasks:
        if task.estimate is None:
            continue
        count = task.estimate.kernel_count_estimate
        graph += count if task.captured else 0
        eager += task.estimate.eager_kernel_count_estimate + (0 if task.captured else count)
    return KernelEstimate(graph, eager)


def _taps(
    binding: TensorDeclaration, facts: BackendCompileFacts, layouts: tuple[HiddenTapLayout, ...], shape: ModelShape
) -> tuple[TapCost, ...]:
    logit_taps = {shape.hidden_binding(layer, HiddenSite.RESIDUAL).tap for layer in facts.probe.layer_logits}
    result: list[TapCost] = []
    for layout in layouts:
        users: set[str] = set()
        for task in binding.graph.tasks:
            for port in task.inputs:
                source = port.source
                if isinstance(source, HiddenInputSource) and (
                    shape.hidden_binding(source.layer, source.site, source.route).tap == layout.tap
                    or (
                        source.site is HiddenSite.LAYER_LOGITS
                        and (
                            (source.route is None and layout.tap in logit_taps)
                            or (
                                source.route is not None
                                and layout.route is not None
                                and layout.site is HiddenSite.RESIDUAL
                                and layout.route.resource == source.route.resource
                                and layout.route.field == source.route.field
                            )
                        )
                    )
                ):
                    users.update(binding.task_members.get(task.symbol, (task.symbol,)))
        result.append(
            TapCost(
                *layout.tap, tuple(phase.value for phase in layout.phases), layout.resident_bytes, tuple(sorted(users))
            )
        )
    return tuple(result)


def _states(plan: PlanBundle, binding: TensorDeclaration, rows: int) -> tuple[StateCost, ...]:
    states = {str(state.resource): state for state in plan.semantic_graph.states}
    layouts = {entry.resource: entry.layout for entry in plan.tensor_layout_request.layouts}
    costs: list[StateCost] = []
    for storage in plan.states.plans:
        state, layout = states[storage.resource], layouts[storage.resource]
        transfer = (
            sum(size for (_, resource), size in binding.gather_bytes.items() if resource == storage.resource)
            if storage.host_bytes
            else 0
        )
        costs.append(
            StateCost(
                storage.resource,
                state.scope.value,
                storage.page_capacities,
                sum(page_stride_bytes(field) for field in layout.fields),
                storage.reserved_bytes,
                storage.max_admitted_lineages,
                layout.fields,
                binding.state_sources.get(storage.resource, "initialize"),
                state.placement,
                storage.host_bytes,
                transfer,
                transfer * rows,
            )
        )
    return tuple(costs)


def _settings(
    binding: TensorDeclaration, config: ModelConfig, plan: PlanBundle, facts: BackendCompileFacts
) -> tuple[EscapeHatch, ...]:
    """The settings that shaped this plan: run-wide options, then each member's and task's choices."""
    lag = config.execution.publish_lag
    result = [
        EscapeHatch(
            "",
            "publish_lag",
            str(lag),
            lag == 1,
            f"updates at step n become readable at n + {1 + lag}; "
            + ("training may overlap later forwards" if lag else "updates serialize before the next forward"),
        )
    ]
    if any(task.sampler_beta for task in binding.tensor_tasks.values()):
        result.append(_guidance_setting(config, facts))
    custom_forward = plan.semantic_graph.custom_forward_task is not None
    for name, member in binding.members.items():
        result.extend(_member_settings(name, member, custom_forward=custom_forward))
    for task in plan.semantic_graph.tasks:
        result.extend(_task_settings(task))
    for resource, field_name, _, over in binding.effects.route_bounds:
        result.append(
            EscapeHatch(
                f"{resource}.{field_name}",
                "LayerRoute",
                f"over={over}",
                False,
                f"device-gated hooks installed on {len(over)} candidate layers (-1 disables); CUDA taps, add/low_rank_affine<=256 patches and dense/low_rank<=256 adapters skip inactive pool reads/GEMMs (patch boundaries still pass norm ports; per_head/rank>256 mask after evaluation)",
            )
        )
    if config.cuda_graphs is False:
        result.append(EscapeHatch("", "cuda_graphs", "False", False, "model forward runs eagerly"))
    return tuple(result)


def _guidance_setting(config: ModelConfig, facts: BackendCompileFacts) -> EscapeHatch:
    """Whether sampler guidance is bounded to the top survivors or exact over the dense vocabulary, and why."""
    capacity = facts.guidance_capacity
    return EscapeHatch(
        "",
        "guidance_capacity",
        str(config.execution.guidance_capacity),
        config.execution.guidance_capacity == ExecutionOptions().guidance_capacity,
        (
            "exact dense post-filter guidance: "
            + (
                "explicit guidance_capacity=None"
                if config.execution.guidance_capacity is None
                else str(facts.guidance_rejection)
            )
        )
        if capacity is None
        else f"bounded fused guidance: guide the highest {capacity} survivors/row, tail unguided; "
        "peaked full-vocabulary rows use the true one-token nucleus when top-1 probability >= top_p < 1; "
        "choose capacities so the common case never takes the overflow path; "
        "smaller K is slower because overflow rows need top-K selection; "
        "unquantized unscaled TP=1 LM head; FLOP/byte estimates are capacity upper bounds",
    )


def _member_settings(name: str, member: MemberDeclaration, *, custom_forward: bool) -> list[EscapeHatch]:
    """A workload member's timing, token and visibility choices, and how its model effect runs."""
    result: list[EscapeHatch] = []
    if member.phase is not None:
        result.append(EscapeHatch(name, "phase", member.phase, False, "overrides automatic task timing"))
    if member.tokens is not None:
        default_tokens = "prefill" if member.effect == "train_in_forward" else "all"
        result.append(
            EscapeHatch(
                name,
                "tokens",
                member.tokens,
                member.tokens == default_tokens,
                "selects the rows this method processes",
            )
        )
    if member.visibility is not None:
        result.append(
            EscapeHatch(
                name,
                "visibility",
                member.visibility,
                member.visibility == "forward_pinned",
                "later chunks read earlier updates"
                if member.visibility == "exact"
                else "forward reads its entry state",
            )
        )
    if member.effect == "custom_forward":
        result.append(
            EscapeHatch(
                name, member.effect, "enabled", False, "model wrapper runs eagerly; layer waves use CUDA graphs"
            )
        )
    elif member.effect == "Adapter" and custom_forward:
        result.append(
            EscapeHatch(
                name,
                "projection calls",
                "forward-pinned state",
                True,
                "apply the delta on every projection invocation, including repeated layer calls and replay; "
                "compacted rows and reordered waves select each request through device indices; "
                "native QKV propagation also uses the delta; 0 blocking host syncs per step",
            )
        )
    elif member.effect == "custom_residual":
        result.append(
            EscapeHatch(
                name, member.effect, "enabled", False, "model graphs capture source operators, not inductor fusion"
            )
        )
    return result


def _task_settings(task: SemanticTask) -> list[EscapeHatch]:
    """A task's adaptive-depth execution and the hidden-row phases it reads."""
    result: list[EscapeHatch] = []
    if task.custom_forward is not None and task.custom_forward.adaptive:
        result.append(
            EscapeHatch(
                task.symbol,
                "active",
                "CUDA conditional nodes",
                True,
                "stable compaction into power-of-two buckets plus exact all-active width; 0 blocking host syncs per step; "
                "first skipped commit propagates native norm/QKV/RoPE, later skips retain KV; recompute replays by token role; "
                "decode GEMMs save most when all rows exit; full-QKV propagation is about 7% of Qwen2.5 layer FLOPs (estimate)",
            )
        )
    phases = tuple(
        dict.fromkeys(
            port.source.phase.value
            for port in task.inputs
            if isinstance(port.source, HiddenInputSource) and port.source.site is not HiddenSite.LAYER_LOGITS
        )
    )
    if phases:
        result.append(
            EscapeHatch(
                task.symbol,
                "tokens",
                ",".join(phases),
                phases == ("decode",),
                "hidden taps select completed prompt endings, latest decode rows or every fresh token",
            )
        )
    return result


def _plans(
    workload: Workload, facts: BackendCompileFacts, binding: TensorDeclaration, config: ModelConfig
) -> tuple[BackendAttachment, int | None, int, tuple[str, ...]]:
    preferred = workload.plan_backend(facts, binding)
    peak = preferred.plan.schedule.peak_memory_bytes
    inline = preferred
    threshold = None
    if peak and any(task.lane == "side" for task in preferred.plan.schedule.tasks):
        try:
            inline = workload.plan_backend(
                replace(facts, timeline=replace(facts.timeline, execution_peak_budget_bytes=peak - 1)), binding
            )
        except ValueError:
            pass  # No cheaper schedule: overlap costs no extra peak memory.
        else:
            threshold = peak
    errors: list[str] = []
    selected = preferred
    options = config.execution
    if options.execution_peak_budget_bytes is not None or options.state_page_budget_bytes:
        limited = replace(
            facts,
            timeline=replace(facts.timeline, execution_peak_budget_bytes=options.execution_peak_budget_bytes),
            state_page_budget_bytes=options.state_page_budget_bytes or facts.state_page_budget_bytes,
        )
        try:
            selected = workload.plan_backend(limited, binding)
        except ValueError as error:
            errors.append(str(error))
    return selected, threshold, inline.plan.schedule.peak_memory_bytes, tuple(errors)


def _specialize(
    workload: Workload, facts: BackendCompileFacts, config: ModelConfig
) -> tuple[BackendCompileFacts, TensorDeclaration, tuple[str, ...]]:
    errors: tuple[str, ...] = ()
    try:
        funded, binding = workload.specialize(facts, config.execution, prefix_caching=config.enable_prefix_caching)
    except ValueError as error:
        if (
            config.execution.execution_peak_budget_bytes is None
            and not config.execution.state_page_budget_bytes
            and config.execution.cuda_graph_pool_budget_bytes is None
        ):
            raise
        # If the declaration is invalid independently of a budget, this raises its own error again.
        funded, binding = workload.specialize(
            facts,
            replace(
                config.execution,
                execution_peak_budget_bytes=None,
                state_page_budget_bytes=0,
                cuda_graph_pool_budget_bytes=None,
            ),
            prefix_caching=config.enable_prefix_caching,
        )
        errors = (str(error),)
    return (
        replace(
            funded,
            timeline=replace(funded.timeline, execution_peak_budget_bytes=None),
            state_page_budget_bytes=facts.state_page_budget_bytes,
        ),
        binding,
        errors,
    )


def explain(workload: Workload, config: ModelConfig) -> ExplainResult:
    """Plan using model/quantization metadata and shape-only tracing; never load weights or allocate CUDA storage.

    Declared pretrained state files are read and validated on CPU.

    Memory excludes base-model weights, KV cache and vLLM model-graph/activation pools. GPU fit is decided by make_llm.
    A too-small declared execution/state budget returns the unrestricted plan with actionable budget_errors.
    """
    from pllm.backends.vllm.admission import admit_workload, model_hooks
    from pllm.backends.vllm.architecture import post_attention_norm_name
    from pllm.backends.vllm.compile_facts import compile_facts
    from pllm.backends.vllm.hidden_taps import HiddenTapAnnouncement
    from pllm.backends.vllm.model_shape import ModelShape

    requirements = workload.backend_requirements()
    reason = admit_workload(config, requirements)
    shape = ModelShape.of_config(
        config,
        requirements.parameter_targets,
        requirements.routed_targets,
        requirements.shared_layer_groups,
    )
    layouts = HiddenTapAnnouncement.declared(requirements, config).resolve(
        shape.layer_count, shape.hidden_size, shape.vocab_size, post_attention_norm_name(shape.architecture)
    )
    facts = compile_facts(
        shape,
        requirements,
        config,
        model_identity=shape.architecture,
        hidden_layouts=layouts,
        overlay_graph_mode=AttachmentGraphMode.CUDA_GRAPH
        if model_hooks(requirements) and not reason
        else AttachmentGraphMode.EAGER,
        memory_bytes=1 << 60,
    )
    return explain_facts(workload, config, shape, facts, layouts)


def explain_facts(
    workload: Workload,
    config: ModelConfig,
    shape: ModelShape,
    facts: BackendCompileFacts,
    layouts: tuple[HiddenTapLayout, ...],
) -> ExplainResult:
    """Render one pure planning computation for both config facts and the loaded GPU probe.

    BLAS scratch is excluded from this CPU-comparable report; make_llm funds it separately.
    """
    from pllm.backends.vllm.admission import admit_workload, eager_reasons
    from pllm.backends.vllm.site_estimates import estimate_sites

    requirements = workload.backend_requirements()
    reason = admit_workload(config, requirements)
    facts = replace(facts, probe=replace(facts.probe, blas_workspace_bytes=0, device_index=0))
    facts, binding, specialization_errors = _specialize(workload, facts, config)
    attachment, threshold, inline_peak, errors = _plans(workload, facts, binding, config)
    plan = attachment.plan
    estimates, ops, model_kernels = estimate_tasks(
        binding, facts, config.execution, padded_vocab_size=shape.padded_vocab_size
    )
    costs = _task_costs(plan, estimates)
    forward_task = binding.graph.custom_forward_task
    model_captured = facts.probe.graph_mode is AttachmentGraphMode.CUDA_GRAPH or (
        forward_task is not None and costs[forward_task.symbol].captured
    )
    sites = {
        name: KernelEstimate(graph, eager) if model_captured else KernelEstimate(0, graph + eager)
        for name, (graph, eager) in estimate_sites(binding, facts).items()
    }
    for name, count in model_kernels.items():
        site = sites.get(name, KernelEstimate())
        sites[name] = (
            KernelEstimate(site.captured + count, site.eager)
            if model_captured
            else KernelEstimate(site.captured, site.eager + count)
        )
    return ExplainResult(
        shape,
        config.dtype,
        config.tensor_parallel_size,
        reason,
        _members(
            binding, costs, dict(eager_reasons(config, requirements)), config.execution.max_host_training_rows, sites
        ),
        tuple(cost for symbol, cost in costs.items() if symbol not in binding.task_members),
        _states(plan, binding, config.max_num_seqs),
        tuple(
            AnchorCost(anchor.symbol, value)
            for anchor, value in zip(plan.timeline.anchors, plan.timeline.reserved_bytes, strict=True)
        ),
        _taps(binding, facts, layouts, shape),
        plan.schedule.peak_memory_bytes,
        "per-rank tensor reservations; excludes vLLM model-graph/activation pools and GPU-dependent BLAS workspace",
        threshold,
        inline_peak,
        (*specialization_errors, *errors),
        preemption_policy(attachment, tuple(binding.sites.rewrites)),
        plan.schedule.recompute_rejection,
        None
        if plan.semantic_graph.trains_on_batch or binding.sites.rewrites or binding.attention.methods or ops
        else 0,
        _exports(binding, facts),
        _settings(binding, config, plan, facts),
        (
            "Hidden taps, route math and PositionState are replicated across TP ranks; pLLM issues no tensor collectives. PositionState uses each rank's KV budget.",
            "GPU fit is decided at make_llm after vLLM allocates; weights, KV cache, vLLM activations and fragmentation are excluded.",
            "Projection and scan BLAS workspace depends on the GPU; affected peaks and overlap thresholds are lower bounds until make_llm.",
            "FLOP and byte estimates use max_num_seqs decode rows, declared-field reads and full-root transitions including optimizer fields, no cache reuse; prompt/model-site costs can differ.",
            "Kernel estimates count non-view device dispatches before fusion at max_num_seqs decode rows; opaque ops count once, libraries may launch several kernels. Graph counts are nodes inside replay, not host launches; state copies and sampler stages are eager. Publications only change ownership metadata. Untraced host-batch and native custom-forward model math are excluded; profile for actual counts.",
            "Host sync count covers shape-traced workload math; model-site math and framework syncs need GPU verification.",
            "Timing settings equal to defaults are labelled default; explicit use of the same value is indistinguishable.",
        ),
        ops,
        binding.attention.costs,
        layout_capacity(shape, config, workload.backend_requirements().kv_layouts) if binding.attention.methods else (),
        _kernel_sum(tuple(costs.values()), tuple(sites.values())),
        _flops_per_token(shape.base_forward_flops_per_token_estimate, tuple(costs.values())),
        requirements.shared_layer_groups,
    )
