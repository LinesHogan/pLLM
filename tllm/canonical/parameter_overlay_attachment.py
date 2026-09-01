"""Python-Rust codec for canonical parameter-overlay attachment bundles."""

from __future__ import annotations

import json
from typing import TypedDict, cast

from tllm import _core
from tllm.canonical.scheduler import (
    encode_schedule_decision,
    encode_schedule_physical_plan,
    encode_scheduler_timeline,
    encode_semantic_graph,
)
from tllm.canonical.state_representation import (
    encode_state_physical_plan,
    encode_state_representation_decision,
    encode_state_representation_request,
)
from tllm.canonical.tensor_layout import (
    encode_tensor_layout_decision,
    encode_tensor_layout_request,
)
from tllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterOverlayOperator,
    ParameterRole,
    ParameterTarget,
)
from tllm.contracts.parameter_overlay_attachment import (
    ParameterOverlayAttachmentBinding,
    ParameterOverlayAttachmentCandidate,
    ParameterOverlayAttachmentDecision,
    ParameterOverlayAttachmentInputs,
    ParameterOverlayAttachmentPhysicalPlan,
    ParameterOverlayAttachmentRequest,
    ParameterOverlayStage,
    ParameterOverlayStageAnchor,
    ParameterOverlayScope,
    ParameterOverlayTaskBinding,
    ParameterOverlayTaskRole,
)
from tllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    ImplementationId,
    TaskId,
)
from tllm.contracts.tensor_layout import LayoutFieldId


class _BindingWire(TypedDict):
    layer: int
    module: str
    role: str
    operator: str
    field_ids: list[int]


class _StageWire(TypedDict):
    stage: str
    anchor_id: int
    anchor_symbol: str


class _TaskWire(TypedDict):
    role: str
    task_id: int
    implementation_id: str
    anchor_id: int


def _decoded(payload: str) -> object:
    return cast(object, json.loads(payload))


def _bindings(
    values: tuple[ParameterOverlayAttachmentBinding, ...],
) -> list[_BindingWire]:
    return [
        _BindingWire(
            layer=item.target.layer,
            module=item.target.module.value,
            role=item.target.role.value,
            operator=item.operator.value,
            field_ids=[int(field) for field in item.fields],
        )
        for item in values
    ]


def _stages(values: tuple[ParameterOverlayStageAnchor, ...]) -> list[_StageWire]:
    return [
        _StageWire(
            stage=item.stage.value,
            anchor_id=int(item.anchor_id),
            anchor_symbol=item.anchor_symbol,
        )
        for item in values
    ]


def _tasks(values: tuple[ParameterOverlayTaskBinding, ...]) -> list[_TaskWire]:
    return [
        _TaskWire(
            role=item.role.value,
            task_id=int(item.task_id),
            implementation_id=str(item.implementation_id),
            anchor_id=int(item.anchor_id),
        )
        for item in values
    ]


def _request(request: ParameterOverlayAttachmentRequest) -> dict[str, object]:
    return {
        "adapter_id": request.adapter_id,
        "backend_abi_digest": request.backend_abi_digest,
        "runner_fingerprint": request.runner_fingerprint,
        "model_fingerprint": request.model_fingerprint,
        "state_resource": request.state_resource,
        "scope": request.scope.value,
        "request_key_fields": [field.value for field in request.request_key_fields],
        "bindings": _bindings(request.bindings),
        "operand_capture": request.operand_capture.value,
        "stage_anchors": _stages(request.stage_anchors),
        "task_bindings": _tasks(request.task_bindings),
        "plan_epoch": request.plan_epoch,
        "graph_signature": request.graph_signature,
        "schedule_signature": request.schedule_signature,
        "state_representation_signature": request.state_representation_signature,
        "capabilities": list(request.capabilities),
        "schema_version": request.schema_version,
    }


def _candidate(candidate: ParameterOverlayAttachmentCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "adapter_id": candidate.adapter_id,
        "backend_abi_digest": candidate.backend_abi_digest,
        "runner_fingerprint": candidate.runner_fingerprint,
        "model_fingerprint": candidate.model_fingerprint,
        "scope": candidate.scope.value,
        "bindings": _bindings(candidate.bindings),
        "operand_capture": candidate.operand_capture.value,
        "stage_anchors": _stages(candidate.stage_anchors),
        "task_bindings": _tasks(candidate.task_bindings),
        "capabilities": list(candidate.capabilities),
    }


def _inputs_json(inputs: ParameterOverlayAttachmentInputs) -> str:
    wire: dict[str, object] = {
        "request": _request(inputs.request),
        "candidates": [_candidate(candidate) for candidate in inputs.candidates],
        "graph": _decoded(encode_semantic_graph(inputs.graph)),
        "timeline": _decoded(encode_scheduler_timeline(inputs.timeline)),
        "schedule_decision": _decoded(
            encode_schedule_decision(inputs.schedule_decision)
        ),
        "schedule_physical": _decoded(
            encode_schedule_physical_plan(inputs.schedule_physical)
        ),
        "state_request": _decoded(
            encode_state_representation_request(inputs.state_request)
        ),
        "state_decision": _decoded(
            encode_state_representation_decision(inputs.state_decision)
        ),
        "state_physical": _decoded(encode_state_physical_plan(inputs.state_physical)),
        "tensor_layout_request": _decoded(encode_tensor_layout_request(inputs.tensor_layout_request)),
        "tensor_layout_decision": _decoded(encode_tensor_layout_decision(inputs.tensor_layout_decision)),
    }
    return json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decision_json(value: ParameterOverlayAttachmentDecision) -> str:
    return json.dumps(
        {
            "request_digest": value.request_digest,
            "selected_candidate_id": value.selected_candidate_id,
            "signature": value.signature,
            "schema_version": value.schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _bindings_from(
    value: object,
) -> tuple[ParameterOverlayAttachmentBinding, ...]:
    if type(value) is not list:
        raise TypeError("attachment bindings must be an array")
    parsed: list[ParameterOverlayAttachmentBinding] = []
    for position, item in enumerate(cast(list[object], value)):
        if type(item) is not dict:
            raise TypeError(f"attachment bindings[{position}] must be an object")
        fields = cast(dict[object, object], item)
        if set(fields) != {"layer", "module", "role", "operator", "field_ids"}:
            raise ValueError(f"attachment bindings[{position}] has wrong fields")
        layer = fields["layer"]
        module = fields["module"]
        role = fields["role"]
        operator = fields["operator"]
        field_ids = fields["field_ids"]
        if (
            type(layer) is not int
            or type(module) is not str
            or type(role) is not str
            or type(operator) is not str
        ):
            raise TypeError(f"attachment bindings[{position}] has invalid types")
        if type(field_ids) is not list or any(
            type(field) is not int for field in cast(list[object], field_ids)
        ):
            raise TypeError(f"attachment bindings[{position}] has invalid fields")
        parsed.append(
            ParameterOverlayAttachmentBinding(
                ParameterTarget(
                    layer,
                    ParameterModuleKind(module),
                    ParameterRole(role),
                ),
                ParameterOverlayOperator(operator),
                tuple(
                    LayoutFieldId(field)
                    for field in cast(list[int], field_ids)
                ),
            )
        )
    return tuple(parsed)


def _stages_from(value: object) -> tuple[ParameterOverlayStageAnchor, ...]:
    if type(value) is not list:
        raise TypeError("attachment stages must be an array")
    parsed: list[ParameterOverlayStageAnchor] = []
    for position, item in enumerate(cast(list[object], value)):
        if type(item) is not dict:
            raise TypeError(f"attachment stages[{position}] must be an object")
        fields = cast(dict[object, object], item)
        if set(fields) != {"stage", "anchor_id", "anchor_symbol"}:
            raise ValueError(f"attachment stages[{position}] has wrong fields")
        stage = fields["stage"]
        anchor_id = fields["anchor_id"]
        anchor_symbol = fields["anchor_symbol"]
        if (
            type(stage) is not str
            or type(anchor_id) is not int
            or type(anchor_symbol) is not str
        ):
            raise TypeError(f"attachment stages[{position}] has invalid types")
        parsed.append(
            ParameterOverlayStageAnchor(
                ParameterOverlayStage(stage),
                AnchorId(anchor_id),
                AnchorSymbol(anchor_symbol),
            )
        )
    return tuple(parsed)


def _tasks_from(value: object) -> tuple[ParameterOverlayTaskBinding, ...]:
    if type(value) is not list:
        raise TypeError("attachment tasks must be an array")
    parsed: list[ParameterOverlayTaskBinding] = []
    for position, item in enumerate(cast(list[object], value)):
        if type(item) is not dict:
            raise TypeError(f"attachment tasks[{position}] must be an object")
        fields = cast(dict[object, object], item)
        if set(fields) != {"role", "task_id", "implementation_id", "anchor_id"}:
            raise ValueError(f"attachment tasks[{position}] has wrong fields")
        role = fields["role"]
        task_id = fields["task_id"]
        implementation_id = fields["implementation_id"]
        anchor_id = fields["anchor_id"]
        if (
            type(role) is not str
            or type(task_id) is not int
            or type(implementation_id) is not str
            or type(anchor_id) is not int
        ):
            raise TypeError(f"attachment tasks[{position}] has invalid types")
        parsed.append(
            ParameterOverlayTaskBinding(
                ParameterOverlayTaskRole(role),
                TaskId(task_id),
                ImplementationId(implementation_id),
                AnchorId(anchor_id),
            )
        )
    return tuple(parsed)


def _decision_from(payload: str) -> ParameterOverlayAttachmentDecision:
    decoded = _decoded(payload)
    if type(decoded) is not dict:
        raise TypeError("parameter-overlay decision must be an object")
    raw = cast(dict[object, object], decoded)
    if set(raw) != {
        "request_digest",
        "selected_candidate_id",
        "signature",
        "schema_version",
    }:
        raise ValueError("parameter-overlay decision has wrong fields")
    request_digest = raw["request_digest"]
    selected_candidate_id = raw["selected_candidate_id"]
    signature = raw["signature"]
    schema_version = raw["schema_version"]
    if (
        type(request_digest) is not str
        or type(selected_candidate_id) is not str
        or type(signature) is not str
        or type(schema_version) is not int
    ):
        raise TypeError("parameter-overlay decision has invalid types")
    return ParameterOverlayAttachmentDecision(
        request_digest,
        selected_candidate_id,
        signature,
        schema_version,
    )


def _physical_from(payload: str) -> ParameterOverlayAttachmentPhysicalPlan:
    decoded = _decoded(payload)
    if type(decoded) is not dict:
        raise TypeError("parameter-overlay physical plan must be an object")
    raw = cast(dict[object, object], decoded)
    if set(raw) != {
        "plan_epoch",
        "selected_candidate_id",
        "attachment_signature",
        "schedule_signature",
        "state_representation_signature",
        "scope",
        "bindings",
        "stage_anchors",
        "task_bindings",
        "bundle_signature",
        "schema_version",
    }:
        raise ValueError("parameter-overlay physical plan has wrong fields")
    plan_epoch = raw["plan_epoch"]
    selected_candidate_id = raw["selected_candidate_id"]
    attachment_signature = raw["attachment_signature"]
    schedule_signature = raw["schedule_signature"]
    state_signature = raw["state_representation_signature"]
    scope = raw["scope"]
    bundle_signature = raw["bundle_signature"]
    schema_version = raw["schema_version"]
    if (
        type(plan_epoch) is not int
        or type(selected_candidate_id) is not str
        or type(attachment_signature) is not str
        or type(schedule_signature) is not str
        or type(state_signature) is not str
        or type(scope) is not str
        or type(bundle_signature) is not str
        or type(schema_version) is not int
    ):
        raise TypeError("parameter-overlay physical plan has invalid types")
    return ParameterOverlayAttachmentPhysicalPlan(
        plan_epoch,
        selected_candidate_id,
        attachment_signature,
        schedule_signature,
        state_signature,
        ParameterOverlayScope(scope),
        _bindings_from(raw["bindings"]),
        _stages_from(raw["stage_anchors"]),
        _tasks_from(raw["task_bindings"]),
        bundle_signature,
        schema_version,
    )


def _physical_json(value: ParameterOverlayAttachmentPhysicalPlan) -> str:
    return json.dumps(
        {
            "plan_epoch": value.plan_epoch,
            "selected_candidate_id": value.selected_candidate_id,
            "attachment_signature": value.attachment_signature,
            "schedule_signature": value.schedule_signature,
            "state_representation_signature": (
                value.state_representation_signature
            ),
            "scope": value.scope.value,
            "bindings": _bindings(value.bindings),
            "stage_anchors": _stages(value.stage_anchors),
            "task_bindings": _tasks(value.task_bindings),
            "bundle_signature": value.bundle_signature,
            "schema_version": value.schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def compile_parameter_overlay_attachment(
    inputs: ParameterOverlayAttachmentInputs,
) -> ParameterOverlayAttachmentDecision:
    return _decision_from(
        _core.compile_parameter_overlay_attachment(_inputs_json(inputs))
    )


def validate_parameter_overlay_attachment(
    inputs: ParameterOverlayAttachmentInputs,
    decision: ParameterOverlayAttachmentDecision,
) -> None:
    _core.validate_parameter_overlay_attachment(_inputs_json(inputs), _decision_json(decision))


def lower_parameter_overlay_attachment(
    inputs: ParameterOverlayAttachmentInputs,
    decision: ParameterOverlayAttachmentDecision,
) -> ParameterOverlayAttachmentPhysicalPlan:
    return _physical_from(
        _core.lower_parameter_overlay_attachment(
            _inputs_json(inputs),
            _decision_json(decision),
        )
    )


def validate_parameter_overlay_attachment_reconstruction(
    inputs: ParameterOverlayAttachmentInputs,
    decision: ParameterOverlayAttachmentDecision,
    physical: ParameterOverlayAttachmentPhysicalPlan,
    local_inputs: ParameterOverlayAttachmentInputs,
    local_physical: ParameterOverlayAttachmentPhysicalPlan,
) -> None:
    _core.validate_parameter_overlay_attachment_reconstruction(
        _inputs_json(inputs),
        _decision_json(decision),
        _physical_json(physical),
        _inputs_json(local_inputs),
        _physical_json(local_physical),
    )


__all__ = [
    "compile_parameter_overlay_attachment",
    "lower_parameter_overlay_attachment",
    "validate_parameter_overlay_attachment",
    "validate_parameter_overlay_attachment_reconstruction",
]
