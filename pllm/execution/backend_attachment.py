"""Effect buffer facts and the declaration checks that turn a planned workload into its attachment."""

from __future__ import annotations

from dataclasses import replace
from math import prod

from pllm.contracts.backend_attachment import (
    TOKEN_CAPTURE_CAPABILITY,
    AttachmentGraphMode,
    BackendAttachment,
    BackendAttachmentSpec,
    ModelProbe,
)
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    AnchorSymbol,
    HiddenInputSource,
    InputTokensInputSource,
    PackedSamplerRowsInputSource,
    SamplerFeedbackInputSource,
    SchedulerTimeline,
    SemanticGraph,
    SemanticOutputKind,
    SemanticTask,
    StateEffectKind,
    StateScope,
    StateSelectionInputSource,
    StepIdInputSource,
    TaskResourceFact,
    TaskResultInputSource,
    TaskSymbol,
    TimingConstraint,
    TokenProgramEffect,
    ValueResourceFact,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType, TensorLayoutRequest
from pllm.execution.parameter_projection import ROWS_PER_BLOCK, projection_blocks, uses_pinned_projection


def attachment_buffer_bytes(
    probe: ModelProbe,
    graph: SemanticGraph,
    layouts: TensorLayoutRequest,
    attachment: BackendAttachmentSpec,
) -> int:
    """Fund effect buffers and pinned projection scratch; excludes model scratch."""
    rows, hidden = probe.max_packed_rows, probe.hidden_size
    width = 8 if probe.tensor_dtype is TensorDType.FLOAT64 else 4
    overlaid = {binding.resource for binding in attachment.overlay_bindings}
    fields = tuple(field for layout in layouts.layouts if layout.resource in overlaid for field in layout.layout.fields)
    # Current rows and two retained writer handoffs share the same bounded token shape.
    buffers = 3 * rows * (graph.input_token_history + 1) * 8
    if graph.input_token_history >= 0:
        requests = rows if probe.causal_domain is None else probe.causal_domain.requests
        buffers += (requests + 2) * 8
    buffers += 1024 if attachment.overlay_bindings else 0
    projection_operands: list[int] = []
    if any(binding.reads_embedding for binding in attachment.overlay_bindings):
        buffers += rows * hidden * width
    pinned = uses_pinned_projection(
        in_forward=graph.updates_in_forward, graphs=probe.graph_mode is AttachmentGraphMode.CUDA_GRAPH
    )
    if attachment.overlay_bindings and probe.device_kind is StateDeviceKind.CUDA and pinned:
        buffers += probe.blas_workspace_bytes
    pinned_rows = min(rows, probe.causal_domain.requests) if graph.updates_in_forward and probe.causal_domain else rows
    decode_updates = graph.updates_in_forward and HiddenPhase.DECODE in attachment.overlay_phases
    for binding in attachment.overlay_bindings:
        if binding.reads_embedding:
            buffers += rows * hidden * width
        buffers += rows  # Scalar prompt-end operands for the largest request set.
        first = binding.fields[0].shape
        if binding.operator is ParameterOverlayOperator.DENSE_ADD:
            outputs, inputs = first
            operands = rows * (width * (inputs + hidden) + 8)
            buffers += rows * width * outputs
            if pinned or graph.custom_forward_task is not None:
                buffers += pinned_rows * (width * (inputs + hidden) + 8)
            if not pinned or graph.updates_in_forward:
                # Exact projections consume each layer's operands immediately.
                # The context retains the previous segment until the next replaces it.
                projection_operands.append(operands)
        elif binding.operator is ParameterOverlayOperator.HEAD_ADD:
            heads, width_per_head, _ = first
            inputs = outputs = heads * width_per_head
            # Owned capture, fp32 input/correction, layout copy, cast and addition.
            buffers += rows * (6 * width * heads * width_per_head + 8)
        else:
            rank, inputs = first
            outputs = binding.fields[1].shape[0]
            buffers += rows * (width * (inputs + rank + 2 * outputs) + 8)
        if probe.causal_domain is not None and pinned:
            blocks = 1 if decode_updates else projection_blocks(pinned_rows, probe.causal_domain.requests)
            buffers += 24 * (blocks * (ROWS_PER_BLOCK + len(binding.fields)) + rows)
            buffers += 2 * blocks * sum(prod(field.shape) * width for field in binding.fields)
            buffers += blocks * ROWS_PER_BLOCK * 2 * width * (inputs + outputs)
            if graph.custom_forward_task is not None and attachment.in_forward is None:
                # Per-call row ownership, inverse span mapping and remapped projection ports.
                buffers += 8 * (3 * rows + blocks * ROWS_PER_BLOCK + probe.causal_domain.requests * len(binding.fields))
    state_layouts = {entry.resource: entry.layout for entry in layouts.layouts}
    for binding in attachment.activation_bindings:
        shape = binding.fields[0].shape
        rank = shape[0] if len(shape) == 2 else 0
        # Mask and exact field slots are stable; graph replays gather row fields from PTTS.
        layout = state_layouts[binding.resource]
        gathered = sum(prod(view.shape) * layout.fields[view.field].dtype.itemsize for view in binding.fields)
        buffers += rows * (width * (2 * hidden + 3 * rank) + 8 * (1 + len(binding.fields)) + gathered)
    for task in graph.tasks:
        match task.custom_forward:
            case TokenProgramEffect():
                reads = {effect.resource for effect in task.effects if effect.kind is StateEffectKind.READ}
                buffers += sum(
                    prod(field.shape) * field.dtype.itemsize
                    for layout in layouts.layouts
                    if layout.resource in reads
                    for field in layout.layout.fields
                )
                buffers += 2 * rows * hidden * width
                buffers += (3 * hidden * width + 16) * len(task.custom_forward.position_states) + 2
                if probe.causal_domain is not None and attachment.in_forward is not None:
                    requests = probe.causal_domain.requests
                    # Two phase signatures at every exact width, plus one ordinary span.
                    buffers += (requests * (requests + 1) + requests) * (
                        3 * sum(prod(field.shape) * width for field in fields) + 1
                    )
            case None:
                pass
    return buffers + sum(sorted(projection_operands, reverse=True)[:2])


def derive_attachment_resources(
    probe: ModelProbe,
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    *,
    capture_memory_bytes: int = 0,
) -> SchedulerTimeline:
    """Fund the tokenwise reference plan through its task and value resource facts.

    Preserve producer task/value facts when the model reads a publication.
    The adapter keeps embeddings and output concatenation in reserved_bytes;
    the custom-forward task pays for its own tensors and recovery lifetimes.
    """
    tasks = tuple(task for task in graph.tasks if isinstance(task.custom_forward, TokenProgramEffect))
    if not tasks:
        return timeline
    if len(tasks) != 1:
        raise ValueError("custom-forward attachment costs need one custom-forward task")
    (task,) = tasks
    # Builder._attach composes the custom forward with its publication; a conditional input group keeps
    # all producer facts, so the complete DAG funds the worst case.
    resident = capture_memory_bytes if TOKEN_CAPTURE_CAPABILITY in probe.capabilities else 0
    workspace = probe.max_packed_rows * probe.hidden_size * 8
    existing = next((fact for fact in timeline.task_resources if fact.task == task.symbol), None)
    if existing is not None:
        task_facts = tuple(
            replace(fact, resident_bytes=max(fact.resident_bytes, resident)) if fact.task == task.symbol else fact
            for fact in timeline.task_resources
        )
    else:
        task_facts = (*timeline.task_resources, TaskResourceFact(task.symbol, workspace, resident))
    if timeline.memory_budget_bytes is not None and (
        max(timeline.reserved_bytes, default=0)
        + sum(fact.workspace_bytes + fact.resident_bytes for fact in task_facts)
        + sum(fact.bytes for fact in timeline.value_resources)
        > timeline.memory_budget_bytes
    ):
        raise ValueError("raise memory_budget_bytes to fund the custom forward's workspace and graphs")
    # The output lives in the original adapter's concatenation reservation;
    # the outer task runner does not allocate a second copy of this value.
    known = {(fact.task, fact.output) for fact in timeline.value_resources}
    values = (
        *timeline.value_resources,
        *tuple(
            ValueResourceFact(task.symbol, output.name, 0)
            for output in task.outputs
            if (task.symbol, output.name) not in known
        ),
    )
    return replace(
        timeline,
        task_resources=tuple(sorted(task_facts, key=lambda fact: fact.task)),
        value_resources=tuple(sorted(values, key=lambda fact: (fact.task, fact.output))),
    )


def _check_attachment(attachment: BackendAttachment) -> None:
    """Reject attachments a workload author can declare but the backend cannot run.

    Only mistakes reachable from a declaration are checked, each for its own effect: export inputs, training inputs,
    the trained states' scope and sampled-token readers. Field shapes and dtypes are checked earlier, where the
    declaration resolves them against the model.
    """
    if attachment.spec.exports:
        _check_exports(attachment)
    if attachment.spec.training:
        _check_training(attachment)


def _check_exports(attachment: BackendAttachment) -> None:
    """Exports, and the helpers whose results they read, observe the step between publication and sampling."""
    tasks = {task.symbol: task for task in attachment.plan.semantic_graph.tasks}
    pending, seen = list(attachment.spec.exports), set[TaskSymbol]()
    while pending:
        symbol = pending.pop()
        if symbol in seen:
            continue
        seen.add(symbol)
        task = tasks[symbol]
        output = SemanticOutputKind.HOST if symbol in attachment.spec.exports else SemanticOutputKind.TEMPORARY
        if not _observes(task, output):
            raise ValueError(
                f"{task.symbol}: an export and the helpers it reads run read-only in Phase.AFTER_FORWARD on hidden "
                "rows, state and results; read the value from a helper without state writes or phase overrides"
            )
        pending.extend(port.source.task for port in task.inputs if isinstance(port.source, TaskResultInputSource))


def _observes(task: SemanticTask, output: SemanticOutputKind) -> bool:
    return (
        all(effect.kind is StateEffectKind.READ for effect in task.effects)
        and all(port.kind is output for port in task.outputs)
        and task.timing == TimingConstraint(AnchorSymbol("state.publish"), AnchorSymbol("sampler.pre_draw"))
        and all(
            isinstance(
                port.source,
                HiddenInputSource
                | InputTokensInputSource
                | StepIdInputSource
                | StateSelectionInputSource
                | TaskResultInputSource
                | PackedSamplerRowsInputSource,
            )
            for port in task.inputs
        )
    )


def _check_training(attachment: BackendAttachment) -> None:
    """Row training reads hidden rows; host-batch training already validates its inputs and state scope."""
    spec, graph = attachment.spec, attachment.plan.semantic_graph
    if not graph.trains_on_batch and not attachment.probe.hidden_resources:
        raise ValueError("train and sampler operations need a hidden input, e.g. `h: torch.Tensor = pllm.hidden(0)`")
    adds_logits = any(sampler.stage is SamplerStage.PACKED_PRE_FILTER_ADD for sampler in spec.sampler)
    scopes = {state.resource: state.scope for state in graph.states}
    tasks = {task.symbol: task for task in graph.tasks}
    for symbol in spec.training:
        trained = next(effect.resource for effect in tasks[symbol].effects if effect.kind is not StateEffectKind.READ)
        if not graph.trains_on_batch and scopes[trained] is not StateScope.REQUEST and not adds_logits:
            raise ValueError(
                f"declare the trained state {trained!r} as pllm.RequestState; only workloads with "
                "@pllm.add_logits train an InstanceState"
            )
    # Planning already places every feedback reader after the draw; what remains is who may read it.
    remaining = set(spec.training)
    for scheduled in attachment.plan.schedule.tasks:
        task = graph.tasks[scheduled.task_id]
        feeds_update = bool(remaining) and (task.symbol in spec.training or _read_only_helper(task))
        if any(isinstance(port.source, SamplerFeedbackInputSource) for port in task.inputs) and not (
            adds_logits and feeds_update
        ):
            raise ValueError(
                f"{task.symbol}: sampled tokens are readable only beside @pllm.add_logits, by the train "
                "operation or by read-only operations that feed it"
            )
        remaining.discard(task.symbol)


def _read_only_helper(task: SemanticTask) -> bool:
    return all(effect.kind is StateEffectKind.READ for effect in task.effects) and all(
        output.kind is SemanticOutputKind.TEMPORARY for output in task.outputs
    )


def bind_backend_attachment(plan: PlanBundle, probe: ModelProbe, spec: BackendAttachmentSpec) -> BackendAttachment:
    attachment = BackendAttachment(spec, probe, plan)
    _check_attachment(attachment)
    return attachment


__all__ = ["bind_backend_attachment"]
