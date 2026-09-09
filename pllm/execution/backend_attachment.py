"""Effect buffer facts and closed Python-Rust binding for signed backend attachments."""

from __future__ import annotations


from pllm import _core
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.contracts.backend_attachment import (
    AttachmentCleanupStage,
    AttachmentKind,
    AttachmentTaskBinding,
    AttachmentTaskRole,
    BackendAttachmentCompilerInputs,
    BackendAttachmentDecision,
    BackendAttachmentPhysicalPlan,
    BackendAttachmentRequest,
    BackendPlacementProbe,
    BackendAttachmentSpec,
    SignedBackendAttachment,
    StreamDependency,
)
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.semantic_graph import AnchorId, ConditionalLayerReplay, SemanticGraph, TaskSymbol, TokenwiseLayerReplay
from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from pllm.contracts.tensor_layout import TensorDType, TensorLayoutRequest


def attachment_buffer_bytes(probe: BackendPlacementProbe, graph: SemanticGraph,
                            layouts: TensorLayoutRequest,
                            attachment: BackendAttachmentSpec | BackendAttachmentRequest) -> int:
    """At least FP32 storage for effect buffers; excludes model/library scratch."""
    rows, hidden = probe.max_packed_rows, probe.hidden_size
    width = 8 if probe.tensor_dtype is TensorDType.FLOAT64 else 4
    fields = {field.field_id: field for layout in layouts.layouts
        if layout.resource == attachment.state_resource for field in layout.layout.fields}
    buffers = 0
    for binding in attachment.overlay_bindings:
        first = fields[binding.fields[0]].shape
        if binding.operator is ParameterOverlayOperator.DENSE_ADD:
            outputs, inputs = first
            # Owned hidden/MLP snapshots survive forward; base weights are borrowed.
            buffers += rows * (width * (inputs + hidden + outputs) + 8)
        else:
            rank, inputs = first
            outputs = fields[binding.fields[1]].shape[0]
            buffers += rows * (width * (inputs + rank + 2 * outputs) + 8)
    for binding in attachment.activation_bindings:
        shape = fields[binding.fields[0]].shape
        rank = shape[0] if len(shape) == 2 else 0
        buffers += rows * (width * (2 * hidden + 3 * rank) + 8)
    for task in graph.tasks:
        match task.replay:
            case ConditionalLayerReplay():
                buffers += rows * (3 * width * hidden + 32) + 16 * width * hidden
            case TokenwiseLayerReplay(layer_count=layers):
                # Embeddings, retained normal outputs/cat, mix/carry, one KV slot per layer.
                buffers += 3 * width * rows * hidden + 16 * width * hidden + layers * (2 * width * hidden + 32)
            case None:
                pass
    return buffers


def compile_backend_attachment(
    inputs: BackendAttachmentCompilerInputs,
) -> BackendAttachmentDecision:
    return decode_boundary(_core.compile_backend_attachment(encode_boundary(inputs)), BackendAttachmentDecision)


def validate_backend_attachment(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
) -> None:
    _core.validate_backend_attachment(encode_boundary(inputs), encode_boundary(decision))


def lower_backend_attachment(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
) -> BackendAttachmentPhysicalPlan:
    return decode_boundary(_core.lower_backend_attachment(encode_boundary(inputs), encode_boundary(decision)), BackendAttachmentPhysicalPlan)


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


def validate_backend_attachment_reconstruction(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
    physical: BackendAttachmentPhysicalPlan,
    local_inputs: BackendAttachmentCompilerInputs,
    local_physical: BackendAttachmentPhysicalPlan,
) -> None:
    _core.validate_backend_attachment_reconstruction(
        encode_boundary(inputs),
        encode_boundary(decision),
        encode_boundary(physical),
        encode_boundary(local_inputs),
        encode_boundary(local_physical),
    )


def bind_attachment_tasks(
    plan_bundle: PlanBundle,
    roles: tuple[tuple[AttachmentTaskRole, TaskSymbol], ...],
) -> tuple[AttachmentTaskBinding, ...]:
    bindings: list[AttachmentTaskBinding] = []
    for role, symbol in roles:
        physical, = (task for task in plan_bundle.schedule_physical.tasks
            if plan_bundle.semantic_graph.tasks[task.task_id].symbol == symbol)
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
    plan: PlanBundle, probe: BackendPlacementProbe, spec: BackendAttachmentSpec,
) -> SignedBackendAttachment:
    anchors = {anchor.symbol: anchor for anchor in plan.timeline.anchors}
    sampler = anchors[spec.sampler_anchor]
    dependencies = (
        StreamDependency.HIDDEN_PRODUCER_TO_COMPUTE, StreamDependency.COMPUTE_TO_SAMPLER,
        StreamDependency.SAMPLER_TO_TRANSITION, StreamDependency.TRANSITION_TO_RELEASE,
    )
    if spec.kind is not AttachmentKind.SAMPLER_TRAINING:
        dependencies = (StreamDependency.TRANSITION_TO_RELEASE,)
    if spec.kind is AttachmentKind.OBSERVATION:
        dependencies = (StreamDependency.HIDDEN_PRODUCER_TO_COMPUTE, StreamDependency.TRANSITION_TO_RELEASE)
    request = BackendAttachmentRequest(
        attachment_kind=spec.kind, packed_row_schema_version=1,
        task_bindings=bind_attachment_tasks(plan, spec.tasks),
        sampler_anchor_id=AnchorId(sampler.ordinal), sampler_anchor_symbol=sampler.symbol,
        stream_dependencies=dependencies, sampler_stage=spec.sampler_stage,
        cleanup_stage=AttachmentCleanupStage.REQUEST_FINISH, state_resource=spec.state_resource,
        overlay_bindings=spec.overlay_bindings, activation_bindings=spec.activation_bindings,
    )
    inputs = BackendAttachmentCompilerInputs(request, probe, plan)
    decision = compile_backend_attachment(inputs)
    return SignedBackendAttachment(inputs, decision, lower_backend_attachment(inputs, decision))


__all__ = [
    "bind_attachment_tasks",
    "bind_backend_attachment",
    "compile_backend_attachment",
    "lower_backend_attachment",
    "validate_backend_attachment",
    "validate_backend_attachment_physical",
    "validate_backend_attachment_reconstruction",
]
