"""Effect buffer facts and closed Python-Rust binding for signed backend attachments."""

from __future__ import annotations

from dataclasses import replace

from pllm import _core
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.region_replay import replay_region_workspace_bytes
from pllm.execution.decoder_region import decoder_region_workspace_bytes
from pllm.execution.tokenwise_region import compile_tokenwise_region, compile_tokenwise_source, tokenwise_attachment_bytes
from pllm.contracts.region_program import TensorType
from pllm.contracts.region_proposal import REGION_STREAMS_CAPABILITY
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
from pllm.contracts.semantic_graph import (
    AnchorId, ConditionalLayerReplay, SchedulerTimeline, SemanticGraph, TaskResourceFact, TaskSymbol,
    TokenwiseLayerReplay, ValueResourceFact,
)
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
            case ConditionalLayerReplay(layers=replayed):
                buffers += len(replayed) * (rows * (3 * width * hidden + 32) + 16 * width * hidden)
                buffers += replay_region_workspace_bytes(hidden, width)
                buffers += decoder_region_workspace_bytes(hidden, width)
            case TokenwiseLayerReplay():
                buffers += tokenwise_attachment_bytes(hidden, width, rows)
            case None:
                pass
    return buffers


def derive_attachment_resources(probe: BackendPlacementProbe, graph: SemanticGraph,
                                timeline: SchedulerTimeline) -> SchedulerTimeline:
    """Fund native region candidates through the original AER workspace facts.

    The current tokenwise attachment is one standalone task. Its adapter keeps
    embeddings and output concatenation in reserved_bytes; the task workspace
    pays for the region's tensors and recovery lifetimes at its scheduled anchor.
    """
    tasks = tuple(task for task in graph.tasks if isinstance(task.replay, TokenwiseLayerReplay))
    if not tasks:
        return timeline
    if len(graph.tasks) != 1 or graph.states:
        raise ValueError("tokenwise attachment costs require the original standalone single-device task")
    task, = tasks
    effect = task.replay
    if not isinstance(effect, TokenwiseLayerReplay) or probe.causal_domain is None:
        raise ValueError("tokenwise attachment requires its original model domain")
    tensor = TensorType((1, probe.hidden_size), "torch." + probe.tensor_dtype.value,
        "cpu" if probe.device_kind.value == "cpu" else f"cuda:{probe.device_ordinal}")
    existing = next((fact for fact in timeline.task_resources if fact.task == task.symbol), None)
    if existing is not None:
        region = compile_tokenwise_source(tensor, probe.max_packed_rows, probe.causal_domain, effect,
            source_imports=probe.region_imports, function_bodies=probe.region_bodies)
        region.within_payload_budget(existing.workspace_bytes)
        task_facts = timeline.task_resources
    else:
        region = compile_tokenwise_region(tensor, probe.max_packed_rows, probe.causal_domain, effect, timeline.region_policy,
            stream_lanes=2 if REGION_STREAMS_CAPABILITY in probe.capabilities else 1,
            source_imports=probe.region_imports, function_bodies=probe.region_bodies)
        maximum, = region.payload_limits
        available = maximum.bytes
        if timeline.memory_budget_bytes is not None:
            available = min(available, timeline.memory_budget_bytes - max(timeline.reserved_bytes, default=0)
                - sum(fact.resident_bytes for fact in timeline.task_resources)
                - sum(fact.bytes for fact in timeline.value_resources))
        funded, = region.within_payload_budget(max(0, available)).payload_limits
        task_facts = (*timeline.task_resources, TaskResourceFact(task.symbol, funded.bytes))
    # The output lives in the original adapter's concatenation reservation;
    # the outer task runner does not allocate a second copy of this value.
    known = {(fact.task, fact.output) for fact in timeline.value_resources}
    values = (*timeline.value_resources, *tuple(ValueResourceFact(task.symbol, output.name, 0)
        for output in task.outputs if (task.symbol, output.name) not in known))
    return replace(timeline, task_resources=tuple(sorted(task_facts, key=lambda fact: fact.task)),
        value_resources=tuple(sorted(values, key=lambda fact: (fact.task, fact.output))))


def compile_backend_attachment(
    inputs: BackendAttachmentCompilerInputs,
) -> BackendAttachmentDecision:
    return decode_boundary(_core.compile_backend_attachment(encode_boundary(inputs)), BackendAttachmentDecision)


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
    "validate_backend_attachment_physical",
]
