"""Effect buffer facts and closed Python-Rust binding for signed backend attachments."""

from __future__ import annotations

from dataclasses import replace
from math import prod

from pllm import _core
from pllm.contracts.backend_attachment import (
    AttachmentCleanupStage,
    AttachmentGraphMode,
    AttachmentTaskBinding,
    AttachmentTaskRole,
    BackendAttachmentCompilerInputs,
    BackendAttachmentDecision,
    BackendAttachmentPhysicalPlan,
    BackendAttachmentRequest,
    BackendAttachmentSpec,
    BackendPlacementProbe,
    SignedBackendAttachment,
)
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from pllm.contracts.region_proposal import TOKEN_CAPTURE_CAPABILITY
from pllm.contracts.semantic_graph import (
    AnchorId,
    SchedulerTimeline,
    SemanticGraph,
    StateEffectKind,
    TaskResourceFact,
    TaskResultInputSource,
    TaskSymbol,
    TokenProgramEffect,
    ValueResourceFact,
)
from pllm.contracts.tensor_layout import TensorDType, TensorLayoutRequest
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.parameter_projection import ROWS_PER_BLOCK, projection_blocks


def attachment_buffer_bytes(
    probe: BackendPlacementProbe,
    graph: SemanticGraph,
    layouts: TensorLayoutRequest,
    attachment: BackendAttachmentSpec | BackendAttachmentRequest,
) -> int:
    """At least FP32 storage for effect buffers; excludes model/library scratch."""
    rows, hidden = probe.max_packed_rows, probe.hidden_size
    width = 8 if probe.tensor_dtype is TensorDType.FLOAT64 else 4
    fields = {
        field.field_id: field
        for layout in layouts.layouts
        if layout.resource == attachment.state_resource
        for field in layout.layout.fields
    }
    buffers = 1024 if attachment.overlay_bindings else 0
    for binding in attachment.overlay_bindings:
        first = fields[binding.fields[0]].shape
        if binding.operator is ParameterOverlayOperator.DENSE_ADD:
            outputs, inputs = first
            # Owned hidden/MLP snapshots survive forward; base weights are borrowed.
            buffers += rows * (width * (inputs + hidden + outputs) + 8)
        elif binding.operator is ParameterOverlayOperator.HEAD_ADD:
            heads, width_per_head, _ = first
            inputs = outputs = heads * width_per_head
            # Owned capture, fp32 input/correction, layout copy, cast and addition.
            buffers += rows * (6 * width * heads * width_per_head + 8)
        else:
            rank, inputs = first
            outputs = fields[binding.fields[1]].shape[0]
            buffers += rows * (width * (inputs + rank + 2 * outputs) + 8)
        if probe.causal_domain is not None and (
            probe.graph_mode is AttachmentGraphMode.CUDA_GRAPH or not graph.updates_in_forward
        ):
            blocks = projection_blocks(rows, probe.causal_domain.requests)
            buffers += 24 * (blocks * (ROWS_PER_BLOCK + len(binding.fields)) + rows)
            buffers += 2 * blocks * sum(prod(fields[field].shape) * width for field in binding.fields)
            buffers += blocks * ROWS_PER_BLOCK * 2 * width * (inputs + outputs)
    for binding in attachment.activation_bindings:
        shape = fields[binding.fields[0]].shape
        rank = shape[0] if len(shape) == 2 else 0
        buffers += rows * (width * (2 * hidden + 3 * rank) + 8)
    for task in graph.tasks:
        match task.replay:
            case TokenProgramEffect():
                buffers += 2 * rows * hidden * width
                buffers += (3 * hidden * width + 16) * len(task.replay.position_states) + 2
                if probe.causal_domain is not None and attachment.overlay_bindings:
                    requests = probe.causal_domain.requests
                    # Two phase signatures at every exact width, plus one ordinary span.
                    buffers += (
                        3
                        * (requests * (requests + 1) + requests)
                        * sum(prod(field.shape) * width for field in fields.values())
                    )
            case None:
                pass
    return buffers


def derive_attachment_resources(
    probe: BackendPlacementProbe,
    graph: SemanticGraph,
    timeline: SchedulerTimeline,
    *,
    layouts: TensorLayoutRequest | None = None,
    attachment: BackendAttachmentSpec | BackendAttachmentRequest | None = None,
    capture_memory_bytes: int = 0,
) -> SchedulerTimeline:
    """Fund the tokenwise reference plan through the original AER workspace facts.

    Preserve producer task/value facts when the model reads a publication.
    The adapter keeps embeddings and output concatenation in reserved_bytes;
    the replay task pays for its own tensors and recovery lifetimes.
    """
    tasks = tuple(task for task in graph.tasks if isinstance(task.replay, TokenProgramEffect))
    if not tasks:
        return timeline
    if len(tasks) != 1:
        raise ValueError("tokenwise attachment costs require one model replay task")
    (task,) = tasks
    for group in graph.input_groups:
        if (
            attachment is None
            or group.resource != attachment.state_resource
            or group.readers != (task.symbol,)
            or group.held_version != "input"
        ):
            raise ValueError("conditional tokenwise costs require the exact signed pinned READ")
        # Both branches use this same model source. Retain all producer facts:
        # the complete DAG funds the worst case, including the fresh encoding.
    if (len(graph.tasks) != 1 or graph.states) and task.effects[0].kind is StateEffectKind.READ:
        if len(task.effects) != 1 or task.effects[0].kind is not StateEffectKind.READ:
            raise ValueError("composed tokenwise costs require an exact publication read")
        (read,) = task.effects
        publication = next((candidate for candidate in graph.tasks if candidate.symbol == read.version), None)
        if (
            publication is None
            or not any(state.resource == read.resource for state in graph.states)
            or not any(
                effect.resource == read.resource and effect.kind is StateEffectKind.PUBLISH
                for effect in publication.effects
            )
            or not any(
                isinstance(port.source, TaskResultInputSource)
                and port.source.task == publication.symbol
                and any(output.name == port.source.output for output in publication.outputs)
                for port in task.inputs
            )
        ):
            raise ValueError("composed tokenwise costs require their declared publication dependency")
    effect = task.replay
    if not isinstance(effect, TokenProgramEffect) or probe.causal_domain is None:
        raise ValueError("tokenwise attachment requires its original model domain")
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
        raise ValueError("increase memory_budget_bytes to fund route workspace and captures")
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


def compile_backend_attachment(
    inputs: BackendAttachmentCompilerInputs,
) -> BackendAttachmentDecision:
    return decode_boundary(_core.compile_backend_attachment(encode_boundary(inputs)), BackendAttachmentDecision)


def lower_backend_attachment(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
) -> BackendAttachmentPhysicalPlan:
    return decode_boundary(
        _core.lower_backend_attachment(encode_boundary(inputs), encode_boundary(decision)),
        BackendAttachmentPhysicalPlan,
    )


def validate_backend_attachment_physical(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
    physical: BackendAttachmentPhysicalPlan,
) -> None:
    _core.validate_backend_attachment_physical(
        encode_boundary(inputs),
        encode_boundary(decision),
        encode_boundary(physical),
    )


def bind_attachment_tasks(
    plan_bundle: PlanBundle,
    roles: tuple[tuple[AttachmentTaskRole, TaskSymbol], ...],
) -> tuple[AttachmentTaskBinding, ...]:
    bindings: list[AttachmentTaskBinding] = []
    for role, symbol in roles:
        (physical,) = (
            task
            for task in plan_bundle.schedule_physical.tasks
            if plan_bundle.semantic_graph.tasks[task.task_id].symbol == symbol
        )
        bindings.append(
            AttachmentTaskBinding(
                role,
                physical.task_id,
                physical.implementation_id,
                physical.anchor_id,
            )
        )
    return tuple(bindings)


def bind_backend_attachment(
    plan: PlanBundle,
    probe: BackendPlacementProbe,
    spec: BackendAttachmentSpec,
) -> SignedBackendAttachment:
    anchors = {anchor.symbol: anchor for anchor in plan.timeline.anchors}
    sampler = anchors[spec.sampler_anchor]
    request = BackendAttachmentRequest(
        attachment_kind=spec.kind,
        packed_row_schema_version=1,
        task_bindings=bind_attachment_tasks(plan, spec.tasks),
        sampler_anchor_id=AnchorId(sampler.ordinal),
        sampler_anchor_symbol=sampler.symbol,
        sampler_stage=spec.sampler_stage,
        cleanup_stage=AttachmentCleanupStage.REQUEST_FINISH,
        state_resource=spec.state_resource,
        overlay_bindings=spec.overlay_bindings,
        activation_bindings=spec.activation_bindings,
    )
    inputs = BackendAttachmentCompilerInputs(request, probe, plan)
    decision = compile_backend_attachment(inputs)
    return SignedBackendAttachment(inputs, decision, lower_backend_attachment(inputs, decision))


__all__ = [
    "bind_attachment_tasks",
    "bind_backend_attachment",
    "compile_backend_attachment",
    "lower_backend_attachment",
    "validate_backend_attachment_physical",
]
