"""Closed Python-Rust codec for canonical backend attachment bundles."""

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
from tllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    AttachmentHiddenResource,
    AttachmentTopology,
    BackendAttachmentCompilerInputs,
    BackendAttachmentDecision,
    BackendAttachmentPhysicalPlan,
    BackendAttachmentRequest,
    BackendPlacementProbe,
)
from tllm.contracts.state_representation import StateDeviceKind


class _HiddenWire(TypedDict):
    layer: str
    layer_kind: str
    site: str
    module_path: str
    phase: str


class _TaskBindingWire(TypedDict):
    role: str
    task_id: int
    implementation_id: str
    anchor_id: int


class _RequestWire(TypedDict):
    adapter_id: str
    attachment_kind: str
    backend_abi_digest: str
    runner_fingerprint: str
    model_fingerprint: str
    topology: str
    graph_mode: str
    device_kind: str
    device_ordinal: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    tensor_dtype: str
    hidden_size: int
    max_packed_rows: int
    packed_row_schema_version: int
    hidden_resources: list[_HiddenWire]
    task_bindings: list[_TaskBindingWire]
    sampler_anchor_id: int
    sampler_anchor_symbol: str
    stream_dependencies: list[str]
    sampler_stage: str
    cleanup_stage: str
    request_key_fields: list[str]
    state_resource: str
    plan_epoch: int
    schedule_signature: str
    state_request_digest: str
    state_representation_signature: str
    execution_parameters_digest: str
    capabilities: list[str]
    schema_version: int


class _ProbeWire(TypedDict):
    adapter_id: str
    backend_abi_digest: str
    runner_fingerprint: str
    model_fingerprint: str
    topology: str
    graph_mode: str
    device_kind: str
    device_ordinal: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    tensor_dtype: str
    hidden_size: int
    max_packed_rows: int
    hidden_resources: list[_HiddenWire]
    capabilities: list[str]
    schema_version: int


def _hidden_wire(
    resources: tuple[AttachmentHiddenResource, ...],
) -> list[_HiddenWire]:
    return [
        _HiddenWire(
            layer=resource.binding.layer,
            layer_kind=resource.binding.layer_kind.value,
            site=resource.binding.site.value,
            module_path=resource.binding.module_path,
            phase=resource.phase.value,
        )
        for resource in resources
    ]


def _request_json(request: BackendAttachmentRequest) -> str:
    wire = _RequestWire(
        adapter_id=request.adapter_id,
        attachment_kind=request.attachment_kind.value,
        backend_abi_digest=request.backend_abi_digest,
        runner_fingerprint=request.runner_fingerprint,
        model_fingerprint=request.model_fingerprint,
        topology=request.topology.value,
        graph_mode=request.graph_mode.value,
        device_kind=request.device_kind.value,
        device_ordinal=request.device_ordinal,
        tensor_parallel_size=request.tensor_parallel_size,
        pipeline_parallel_size=request.pipeline_parallel_size,
        tensor_dtype=request.tensor_dtype.value,
        hidden_size=request.hidden_size,
        max_packed_rows=request.max_packed_rows,
        packed_row_schema_version=request.packed_row_schema_version,
        hidden_resources=_hidden_wire(request.hidden_resources),
        task_bindings=[
            _TaskBindingWire(
                role=binding.role.value,
                task_id=int(binding.task_id),
                implementation_id=str(binding.implementation_id),
                anchor_id=int(binding.anchor_id),
            )
            for binding in request.task_bindings
        ],
        sampler_anchor_id=int(request.sampler_anchor_id),
        sampler_anchor_symbol=request.sampler_anchor_symbol,
        stream_dependencies=[
            dependency.value for dependency in request.stream_dependencies
        ],
        sampler_stage=request.sampler_stage.value,
        cleanup_stage=request.cleanup_stage.value,
        request_key_fields=[field.value for field in request.request_key_fields],
        state_resource=request.state_resource,
        plan_epoch=request.plan_epoch,
        schedule_signature=request.schedule_signature,
        state_request_digest=request.state_request_digest,
        state_representation_signature=request.state_representation_signature,
        execution_parameters_digest=request.execution_parameters_digest,
        capabilities=list(request.capabilities),
        schema_version=request.schema_version,
    )
    return json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _probe_json(probe: BackendPlacementProbe) -> str:
    wire = _ProbeWire(
        adapter_id=probe.adapter_id,
        backend_abi_digest=probe.backend_abi_digest,
        runner_fingerprint=probe.runner_fingerprint,
        model_fingerprint=probe.model_fingerprint,
        topology=probe.topology.value,
        graph_mode=probe.graph_mode.value,
        device_kind=probe.device_kind.value,
        device_ordinal=probe.device_ordinal,
        tensor_parallel_size=probe.tensor_parallel_size,
        pipeline_parallel_size=probe.pipeline_parallel_size,
        tensor_dtype=probe.tensor_dtype.value,
        hidden_size=probe.hidden_size,
        max_packed_rows=probe.max_packed_rows,
        hidden_resources=_hidden_wire(probe.hidden_resources),
        capabilities=list(probe.capabilities),
        schema_version=probe.schema_version,
    )
    return json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decoded_json(payload: str) -> object:
    return cast(object, json.loads(payload))


def _inputs_json(inputs: BackendAttachmentCompilerInputs) -> str:
    wire: dict[str, object] = {
        "request": _decoded_json(_request_json(inputs.request)),
        "probe": _decoded_json(_probe_json(inputs.probe)),
        "graph": _decoded_json(encode_semantic_graph(inputs.graph)),
        "timeline": _decoded_json(encode_scheduler_timeline(inputs.timeline)),
        "schedule_decision": _decoded_json(
            encode_schedule_decision(inputs.schedule_decision)
        ),
        "schedule_physical": _decoded_json(
            encode_schedule_physical_plan(inputs.schedule_physical)
        ),
        "state_request": _decoded_json(
            encode_state_representation_request(inputs.state_request)
        ),
        "state_decision": _decoded_json(
            encode_state_representation_decision(inputs.state_decision)
        ),
        "state_physical": _decoded_json(
            encode_state_physical_plan(inputs.state_physical)
        ),
    }
    return json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _decision_json(decision: BackendAttachmentDecision) -> str:
    return json.dumps(
        {
            "request_digest": decision.request_digest,
            "signature": decision.signature,
            "schema_version": decision.schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _decision_from_boundary(payload: str) -> BackendAttachmentDecision:
    decoded = _decoded_json(payload)
    if type(decoded) is not dict:
        raise TypeError("backend attachment decision must be an object")
    fields = cast(dict[object, object], decoded)
    if set(fields) != {"request_digest", "signature", "schema_version"}:
        raise ValueError("backend attachment decision has the wrong fields")
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
    return json.dumps(
        {
            "plan_epoch": physical.plan_epoch,
            "adapter_id": physical.adapter_id,
            "attachment_signature": physical.attachment_signature,
            "schedule_signature": physical.schedule_signature,
            "state_representation_signature": physical.state_representation_signature,
            "device_kind": physical.device_kind.value,
            "device_ordinal": physical.device_ordinal,
            "topology": physical.topology.value,
            "graph_mode": physical.graph_mode.value,
            "bundle_signature": physical.bundle_signature,
            "schema_version": physical.schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _physical_from_boundary(payload: str) -> BackendAttachmentPhysicalPlan:
    decoded = _decoded_json(payload)
    if type(decoded) is not dict:
        raise TypeError("physical backend attachment must be an object")
    fields = cast(dict[object, object], decoded)
    expected = {
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
    }
    if set(fields) != expected:
        raise ValueError("physical backend attachment has the wrong fields")
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


__all__ = [
    "compile_backend_attachment",
    "lower_backend_attachment",
    "validate_backend_attachment",
    "validate_backend_attachment_physical",
    "validate_backend_attachment_reconstruction",
]
