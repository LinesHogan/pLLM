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
from tllm.contracts.request_identity import (
    RequestKeyField,
    canonical_request_key_fields,
)
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

    def __post_init__(self) -> None:
        canonical_request_key_fields(self.request_key_fields)
        if (
            type(self.compute_task) is not str
            or not self.compute_task
            or type(self.transition_task) is not str
            or not self.transition_task
            or self.compute_task == self.transition_task
            or type(self.sampler_anchor) is not str
            or not self.sampler_anchor
            or type(self.state_resource) is not str
            or not self.state_resource
            or type(self.execution_parameters_digest) is not str
            or len(self.execution_parameters_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.execution_parameters_digest
            )
        ):
            raise ValueError("request-sampler attachment spec is not closed")


@dataclass(frozen=True, slots=True)
class SignedBackendAttachment:
    """Full compiler inputs plus their Rust decision and physical projection."""

    inputs: BackendAttachmentCompilerInputs
    decision: BackendAttachmentDecision
    physical: BackendAttachmentPhysicalPlan

    def __post_init__(self) -> None:
        if (
            type(self.inputs) is not BackendAttachmentCompilerInputs
            or type(self.decision) is not BackendAttachmentDecision
            or type(self.physical) is not BackendAttachmentPhysicalPlan
        ):
            raise TypeError("signed backend attachment requires exact artifacts")


def compile_request_sampler_attachment(
    plan_bundle: CanonicalPlanBundle,
    probe: BackendPlacementProbe,
    spec: RequestSamplerAttachmentSpec,
) -> SignedBackendAttachment:
    """Mechanically bind semantic task roles to already-signed physical tasks."""

    if (
        type(plan_bundle) is not CanonicalPlanBundle
        or type(probe) is not BackendPlacementProbe
        or type(spec) is not RequestSamplerAttachmentSpec
    ):
        raise TypeError("request-sampler lowering requires exact compiler facts")

    compute_scheduled = tuple(
        task
        for task in plan_bundle.schedule_decision.tasks
        if task.task_symbol == spec.compute_task
    )
    transition_scheduled = tuple(
        task
        for task in plan_bundle.schedule_decision.tasks
        if task.task_symbol == spec.transition_task
    )
    sampler_anchors = tuple(
        anchor
        for anchor in plan_bundle.timeline.anchors
        if anchor.symbol == spec.sampler_anchor
    )
    if (
        len(compute_scheduled) != 1
        or len(transition_scheduled) != 1
        or len(sampler_anchors) != 1
    ):
        raise ValueError("request-sampler roles do not resolve uniquely")
    compute_physical = tuple(
        task
        for task in plan_bundle.schedule_physical.tasks
        if task.task_id == compute_scheduled[0].task_id
    )
    transition_physical = tuple(
        task
        for task in plan_bundle.schedule_physical.tasks
        if task.task_id == transition_scheduled[0].task_id
    )
    if len(compute_physical) != 1 or len(transition_physical) != 1:
        raise ValueError("request-sampler tasks are absent from the physical plan")

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
        task_bindings=(
            AttachmentTaskBinding(
                AttachmentTaskRole.COMPUTE,
                compute_physical[0].task_id,
                compute_physical[0].implementation_id,
                compute_physical[0].anchor_id,
            ),
            AttachmentTaskBinding(
                AttachmentTaskRole.TRANSITION,
                transition_physical[0].task_id,
                transition_physical[0].implementation_id,
                transition_physical[0].anchor_id,
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
    "compile_request_sampler_attachment",
]
