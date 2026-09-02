"""Typed lowering from request-sampler semantics to one Rust-signed attachment."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.canonical.backend_attachment import (
    compile_backend_attachment,
    lower_backend_attachment,
)
from tllm.contracts.backend_attachment import (
    AttachmentCleanupStage,
    AttachmentKind,
    AttachmentSamplerStage,
    AttachmentTaskBinding,
    AttachmentTaskRole,
    BackendAttachmentCompilerInputs,
    BackendAttachmentDecision,
    BackendAttachmentPhysicalPlan,
    BackendAttachmentRequest,
    BackendPlacementProbe,
    StreamDependency,
)
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.request_identity import RequestKeyField
from tllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    StateResource,
    TaskSymbol,
)


@dataclass(frozen=True, slots=True)
class RequestSamplerAttachmentSpec:
    """Workload semantics needed to bind generic request-sampler execution."""

    compute_task: TaskSymbol
    transition_task: TaskSymbol
    sampler_anchor: AnchorSymbol
    state_resource: StateResource
    request_key_fields: tuple[RequestKeyField, ...]
    execution_parameters_digest: str


@dataclass(frozen=True, slots=True)
class SignedBackendAttachment:
    """Full compiler inputs plus their Rust decision and physical projection."""

    inputs: BackendAttachmentCompilerInputs
    decision: BackendAttachmentDecision
    physical: BackendAttachmentPhysicalPlan


def bind_attachment_tasks(
    plan_bundle: CanonicalPlanBundle,
    roles: tuple[tuple[AttachmentTaskRole, TaskSymbol], ...],
) -> tuple[AttachmentTaskBinding, ...]:
    physical_by_id = {task.task_id: task for task in plan_bundle.schedule_physical.tasks}
    bindings: list[AttachmentTaskBinding] = []
    for role, symbol in roles:
        scheduled = tuple(
            task
            for task in plan_bundle.schedule_decision.tasks
            if task.task_symbol == symbol
        )
        if len(scheduled) != 1 or scheduled[0].task_id not in physical_by_id:
            raise ValueError("attachment task role does not resolve uniquely")
        physical = physical_by_id[scheduled[0].task_id]
        bindings.append(
            AttachmentTaskBinding(
                role,
                physical.task_id,
                physical.implementation_id,
                physical.anchor_id,
            )
        )
    return tuple(bindings)


def compile_request_sampler_attachment(
    plan_bundle: CanonicalPlanBundle,
    probe: BackendPlacementProbe,
    spec: RequestSamplerAttachmentSpec,
) -> SignedBackendAttachment:
    """Mechanically bind semantic task roles to already-signed physical tasks."""

    sampler_anchors = tuple(
        anchor
        for anchor in plan_bundle.timeline.anchors
        if anchor.symbol == spec.sampler_anchor
    )
    if len(sampler_anchors) != 1:
        raise ValueError("request-sampler roles do not resolve uniquely")

    request = BackendAttachmentRequest(
        adapter_id=probe.adapter_id,
        attachment_kind=AttachmentKind.REQUEST_SAMPLER_TRAINING,
        backend_abi_digest=probe.backend_abi_digest,
        runner_fingerprint=probe.runner_fingerprint,
        model_fingerprint=probe.model_fingerprint,
        topology=probe.topology,
        graph_mode=probe.graph_mode,
        device_kind=probe.device_kind,
        device_ordinal=probe.device_ordinal,
        tensor_parallel_size=probe.tensor_parallel_size,
        pipeline_parallel_size=probe.pipeline_parallel_size,
        tensor_dtype=probe.tensor_dtype,
        hidden_size=probe.hidden_size,
        max_packed_rows=probe.max_packed_rows,
        packed_row_schema_version=1,
        hidden_resources=probe.hidden_resources,
        task_bindings=bind_attachment_tasks(
            plan_bundle,
            (
                (AttachmentTaskRole.COMPUTE, spec.compute_task),
                (AttachmentTaskRole.TRANSITION, spec.transition_task),
            ),
        ),
        sampler_anchor_id=AnchorId(sampler_anchors[0].ordinal),
        sampler_anchor_symbol=sampler_anchors[0].symbol,
        stream_dependencies=(
            StreamDependency.HIDDEN_PRODUCER_TO_COMPUTE,
            StreamDependency.COMPUTE_TO_SAMPLER,
            StreamDependency.SAMPLER_TO_TRANSITION,
            StreamDependency.TRANSITION_TO_RELEASE,
        ),
        sampler_stage=AttachmentSamplerStage.PACKED_POST_FILTER_EXACT,
        cleanup_stage=AttachmentCleanupStage.REQUEST_FINISH,
        request_key_fields=spec.request_key_fields,
        state_resource=spec.state_resource,
        plan_epoch=plan_bundle.schedule_physical.plan_epoch,
        schedule_signature=plan_bundle.schedule_physical.schedule_signature,
        state_request_digest=plan_bundle.state_physical.request_digest,
        state_representation_signature=(
            plan_bundle.state_physical.representation_signature
        ),
        execution_parameters_digest=spec.execution_parameters_digest,
        capabilities=probe.capabilities,
    )
    inputs = BackendAttachmentCompilerInputs(
        request=request,
        probe=probe,
        graph=plan_bundle.semantic_graph,
        timeline=plan_bundle.timeline,
        schedule_decision=plan_bundle.schedule_decision,
        schedule_physical=plan_bundle.schedule_physical,
        state_request=plan_bundle.state_request,
        state_decision=plan_bundle.state_decision,
        state_physical=plan_bundle.state_physical,
    )
    decision = compile_backend_attachment(inputs)
    physical = lower_backend_attachment(inputs, decision)
    return SignedBackendAttachment(inputs, decision, physical)


__all__ = [
    "RequestSamplerAttachmentSpec",
    "SignedBackendAttachment",
    "bind_attachment_tasks",
    "compile_request_sampler_attachment",
]
