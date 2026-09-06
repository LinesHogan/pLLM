"""Closed Python-Rust codec for canonical backend attachment bundles."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import cast

from tllm import _core
from tllm.canonical.scheduler import encode_semantic_graph
from tllm.contracts.backend_attachment import (
    AttachmentCleanupStage,
    AttachmentGraphMode,
    AttachmentKind,
    AttachmentSamplerStage,
    AttachmentTaskBinding,
    AttachmentTaskRole,
    AttachmentTopology,
    BackendAttachmentCompilerInputs,
    BackendAttachmentDecision,
    BackendAttachmentPhysicalPlan,
    BackendAttachmentRequest,
    BackendPlacementProbe,
    BackendAttachmentSpec,
    SignedBackendAttachment,
    StreamDependency,
)
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.semantic_graph import AnchorId, TaskSymbol
from tllm.contracts.state_representation import StateDeviceKind


def _decoded_json(payload: str) -> object:
    return cast(object, json.loads(payload))


def _boundary_fields(
    payload: str, expected: set[str], label: str
) -> dict[object, object]:
    decoded = _decoded_json(payload)
    if type(decoded) is not dict:
        raise TypeError(f"{label} must be an object")
    fields = cast(dict[object, object], decoded)
    if set(fields) != expected:
        raise ValueError(f"{label} has the wrong fields")
    return fields


def _inputs_json(inputs: BackendAttachmentCompilerInputs) -> str:
    wire = asdict(inputs)
    wire["plan"]["semantic_graph"] = _decoded_json(encode_semantic_graph(inputs.plan.semantic_graph))
    return json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decision_json(decision: BackendAttachmentDecision) -> str:
    return json.dumps(asdict(decision), sort_keys=True, separators=(",", ":"))


def _decision_from_boundary(payload: str) -> BackendAttachmentDecision:
    fields = _boundary_fields(
        payload,
        {"request_digest", "signature", "schema_version"},
        "backend attachment decision",
    )
    request_digest = fields["request_digest"]
    signature = fields["signature"]
    schema_version = fields["schema_version"]
    if (
        type(request_digest) is not str
        or type(signature) is not str
        or type(schema_version) is not int
    ):
        raise TypeError("backend attachment decision has invalid field types")
    return BackendAttachmentDecision(request_digest, signature, schema_version)


def _physical_json(physical: BackendAttachmentPhysicalPlan) -> str:
    return json.dumps(asdict(physical), sort_keys=True, separators=(",", ":"))


def _physical_from_boundary(payload: str) -> BackendAttachmentPhysicalPlan:
    fields = _boundary_fields(
        payload,
        {
            "plan_epoch",
            "adapter_id",
            "attachment_signature",
            "schedule_signature",
            "state_representation_signature",
            "device_kind",
            "device_ordinal",
            "topology",
            "graph_mode",
            "bundle_signature",
            "schema_version",
        },
        "physical backend attachment",
    )
    plan_epoch = fields["plan_epoch"]
    adapter_id = fields["adapter_id"]
    attachment_signature = fields["attachment_signature"]
    schedule_signature = fields["schedule_signature"]
    state_signature = fields["state_representation_signature"]
    device_kind = fields["device_kind"]
    device_ordinal = fields["device_ordinal"]
    topology = fields["topology"]
    graph_mode = fields["graph_mode"]
    bundle_signature = fields["bundle_signature"]
    schema_version = fields["schema_version"]
    if (
        type(plan_epoch) is not int
        or type(adapter_id) is not str
        or type(attachment_signature) is not str
        or type(schedule_signature) is not str
        or type(state_signature) is not str
        or type(device_kind) is not str
        or type(device_ordinal) is not int
        or type(topology) is not str
        or type(graph_mode) is not str
        or type(bundle_signature) is not str
        or type(schema_version) is not int
    ):
        raise TypeError("physical backend attachment has invalid field types")
    return BackendAttachmentPhysicalPlan(
        plan_epoch=plan_epoch,
        adapter_id=adapter_id,
        attachment_signature=attachment_signature,
        schedule_signature=schedule_signature,
        state_representation_signature=state_signature,
        device_kind=StateDeviceKind(device_kind),
        device_ordinal=device_ordinal,
        topology=AttachmentTopology(topology),
        graph_mode=AttachmentGraphMode(graph_mode),
        bundle_signature=bundle_signature,
        schema_version=schema_version,
    )


def compile_backend_attachment(
    inputs: BackendAttachmentCompilerInputs,
) -> BackendAttachmentDecision:
    return _decision_from_boundary(
        _core.compile_backend_attachment(_inputs_json(inputs))
    )


def validate_backend_attachment(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
) -> None:
    _core.validate_backend_attachment(_inputs_json(inputs), _decision_json(decision))


def lower_backend_attachment(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
) -> BackendAttachmentPhysicalPlan:
    return _physical_from_boundary(
        _core.lower_backend_attachment(_inputs_json(inputs), _decision_json(decision))
    )


def validate_backend_attachment_physical(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
    physical: BackendAttachmentPhysicalPlan,
) -> None:
    _core.validate_backend_attachment_physical(
        _inputs_json(inputs),
        _decision_json(decision),
        _physical_json(physical),
    )


def validate_backend_attachment_reconstruction(
    inputs: BackendAttachmentCompilerInputs,
    decision: BackendAttachmentDecision,
    physical: BackendAttachmentPhysicalPlan,
    local_inputs: BackendAttachmentCompilerInputs,
    local_physical: BackendAttachmentPhysicalPlan,
) -> None:
    _core.validate_backend_attachment_reconstruction(
        _inputs_json(inputs),
        _decision_json(decision),
        _physical_json(physical),
        _inputs_json(local_inputs),
        _physical_json(local_physical),
    )


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


def bind_backend_attachment(
    plan: CanonicalPlanBundle, probe: BackendPlacementProbe, spec: BackendAttachmentSpec,
) -> SignedBackendAttachment:
    anchors = {anchor.symbol: anchor for anchor in plan.timeline.anchors}
    sampler = anchors[spec.sampler_anchor]
    dependencies = (
        StreamDependency.HIDDEN_PRODUCER_TO_COMPUTE, StreamDependency.COMPUTE_TO_SAMPLER,
        StreamDependency.SAMPLER_TO_TRANSITION, StreamDependency.TRANSITION_TO_RELEASE,
    )
    if spec.kind is AttachmentKind.PARAMETER_OVERLAY:
        dependencies = (StreamDependency.TRANSITION_TO_RELEASE,)
    request = BackendAttachmentRequest(
        attachment_kind=spec.kind, packed_row_schema_version=1,
        task_bindings=bind_attachment_tasks(plan, spec.tasks),
        sampler_anchor_id=AnchorId(sampler.ordinal), sampler_anchor_symbol=sampler.symbol,
        stream_dependencies=dependencies, sampler_stage=AttachmentSamplerStage.PACKED_POST_FILTER_EXACT,
        cleanup_stage=AttachmentCleanupStage.REQUEST_FINISH, state_resource=spec.state_resource,
        overlay_bindings=spec.overlay_bindings,
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
