"""Closed FFI and mechanical physical projection for state providers."""

from __future__ import annotations

import json
from typing import TypedDict, cast

from tllm import _core
from tllm.contracts.state_ref import LayoutId
from tllm.contracts.state_representation import (
    StatePhysicalPlan,
    StateProviderPlan,
    StateRepresentation,
    StateRepresentationDecision,
    StateRepresentationRequest,
    encode_state_provider_plans,
)


class _RequirementWire(TypedDict):
    resource: str
    layout_id: int
    layout_namespace: str
    layout_digest: str
    max_admitted_lineages: int
    required_capabilities: list[str]


class _ProviderWire(TypedDict):
    resource: str
    provider_id: str
    representation: str
    layout_id: int
    layout_namespace: str
    layout_digest: str
    max_admitted_lineages: int
    max_live_versions_per_lineage: int
    root_page_count: int
    reserved_bytes: int
    capture_stable: bool
    capabilities: list[str]


class _RequestWire(TypedDict):
    memory_budget_bytes: int
    states: list[_RequirementWire]
    candidates: list[_ProviderWire]
    schema_version: int


def _provider_wire(plan: StateProviderPlan) -> _ProviderWire:
    return _ProviderWire(
        resource=plan.resource,
        provider_id=plan.provider_id,
        representation=plan.representation.value,
        layout_id=int(plan.layout_id),
        layout_namespace=plan.layout_namespace,
        layout_digest=plan.layout_digest,
        max_admitted_lineages=plan.max_admitted_lineages,
        max_live_versions_per_lineage=plan.max_live_versions_per_lineage,
        root_page_count=plan.root_page_count,
        reserved_bytes=plan.reserved_bytes,
        capture_stable=plan.capture_stable,
        capabilities=list(plan.capabilities),
    )


def _request_json(request: StateRepresentationRequest) -> str:
    wire = _RequestWire(
        memory_budget_bytes=request.memory_budget_bytes,
        states=[
            _RequirementWire(
                resource=state.resource,
                layout_id=int(state.layout_id),
                layout_namespace=state.layout_namespace,
                layout_digest=state.layout_digest,
                max_admitted_lineages=state.max_admitted_lineages,
                required_capabilities=list(state.required_capabilities),
            )
            for state in request.states
        ],
        candidates=[_provider_wire(candidate) for candidate in request.candidates],
        schema_version=request.schema_version,
    )
    return json.dumps(
        wire,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _provider_from_boundary(raw: object, label: str) -> StateProviderPlan:
    if type(raw) is not dict:
        raise TypeError(f"{label} must be an object")
    fields = cast(dict[object, object], raw)
    expected = {
        "resource",
        "provider_id",
        "representation",
        "layout_id",
        "layout_namespace",
        "layout_digest",
        "max_admitted_lineages",
        "max_live_versions_per_lineage",
        "root_page_count",
        "reserved_bytes",
        "capture_stable",
        "capabilities",
    }
    if set(fields) != expected:
        raise ValueError(f"{label} has the wrong fields")
    resource = fields["resource"]
    provider_id = fields["provider_id"]
    representation = fields["representation"]
    layout_id = fields["layout_id"]
    layout_namespace = fields["layout_namespace"]
    layout_digest = fields["layout_digest"]
    max_admitted = fields["max_admitted_lineages"]
    max_live = fields["max_live_versions_per_lineage"]
    root_page_count = fields["root_page_count"]
    reserved_bytes = fields["reserved_bytes"]
    capture_stable = fields["capture_stable"]
    capabilities = fields["capabilities"]
    if type(capabilities) is not list:
        raise TypeError(f"{label}.capabilities must be an array")
    capability_values = cast(list[object], capabilities)
    if (
        type(resource) is not str
        or type(provider_id) is not str
        or type(representation) is not str
        or type(layout_id) is not int
        or type(layout_namespace) is not str
        or type(layout_digest) is not str
        or type(max_admitted) is not int
        or type(max_live) is not int
        or type(root_page_count) is not int
        or type(reserved_bytes) is not int
        or type(capture_stable) is not bool
        or any(
            type(capability) is not str
            for capability in capability_values
        )
    ):
        raise TypeError(f"{label} has invalid field types")
    return StateProviderPlan(
        resource=resource,
        provider_id=provider_id,
        representation=StateRepresentation(representation),
        layout_id=LayoutId(layout_id),
        layout_namespace=layout_namespace,
        layout_digest=layout_digest,
        max_admitted_lineages=max_admitted,
        max_live_versions_per_lineage=max_live,
        root_page_count=root_page_count,
        reserved_bytes=reserved_bytes,
        capture_stable=capture_stable,
        capabilities=tuple(cast(list[str], capability_values)),
    )


def _decision_from_boundary(payload: str) -> StateRepresentationDecision:
    decoded = cast(object, json.loads(payload))
    if type(decoded) is not dict:
        raise TypeError("state representation decision must be an object")
    fields = cast(dict[object, object], decoded)
    if set(fields) != {
        "request_digest",
        "plans",
        "reserved_bytes",
        "signature",
        "schema_version",
    }:
        raise ValueError("state representation decision has the wrong fields")
    request_digest = fields["request_digest"]
    plans = fields["plans"]
    reserved_bytes = fields["reserved_bytes"]
    signature = fields["signature"]
    schema_version = fields["schema_version"]
    if (
        type(request_digest) is not str
        or type(plans) is not list
        or type(reserved_bytes) is not int
        or type(signature) is not str
        or type(schema_version) is not int
    ):
        raise TypeError("state representation decision has invalid field types")
    return StateRepresentationDecision(
        request_digest=request_digest,
        plans=tuple(
            _provider_from_boundary(plan, f"plans[{position}]")
            for position, plan in enumerate(cast(list[object], plans))
        ),
        reserved_bytes=reserved_bytes,
        signature=signature,
        schema_version=schema_version,
    )


def compile_state_representation(
    request: StateRepresentationRequest,
) -> StateRepresentationDecision:
    return _decision_from_boundary(
        _core.compile_state_representation_plan(_request_json(request))
    )


def validate_state_representation(
    request: StateRepresentationRequest,
    decision: StateRepresentationDecision,
) -> None:
    _core.validate_state_representation_plan(
        _request_json(request),
        _decision_json(decision),
    )


def validate_state_representation_reconstruction(
    request: StateRepresentationRequest,
    decision: StateRepresentationDecision,
    local_candidates: tuple[StateProviderPlan, ...],
) -> None:
    _core.validate_state_representation_reconstruction(
        _request_json(request),
        _decision_json(decision),
        encode_state_provider_plans(local_candidates),
    )


def _decision_json(decision: StateRepresentationDecision) -> str:
    return json.dumps(
        {
            "request_digest": decision.request_digest,
            "plans": [_provider_wire(plan) for plan in decision.plans],
            "reserved_bytes": decision.reserved_bytes,
            "signature": decision.signature,
            "schema_version": decision.schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _physical_json(physical: StatePhysicalPlan) -> str:
    return json.dumps(
        {
            "plan_epoch": physical.plan_epoch,
            "request_digest": physical.request_digest,
            "representation_signature": physical.representation_signature,
            "representations": [
                _provider_wire(plan) for plan in physical.representations
            ],
            "reserved_bytes": physical.reserved_bytes,
            "schema_version": physical.schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _physical_from_boundary(payload: str) -> StatePhysicalPlan:
    decoded = cast(object, json.loads(payload))
    if type(decoded) is not dict:
        raise TypeError("physical state plan must be an object")
    fields = cast(dict[object, object], decoded)
    if set(fields) != {
        "plan_epoch",
        "request_digest",
        "representation_signature",
        "representations",
        "reserved_bytes",
        "schema_version",
    }:
        raise ValueError("physical state plan has the wrong fields")
    plan_epoch = fields["plan_epoch"]
    request_digest = fields["request_digest"]
    representation_signature = fields["representation_signature"]
    representations = fields["representations"]
    reserved_bytes = fields["reserved_bytes"]
    schema_version = fields["schema_version"]
    if (
        type(plan_epoch) is not int
        or type(request_digest) is not str
        or type(representation_signature) is not str
        or type(representations) is not list
        or type(reserved_bytes) is not int
        or type(schema_version) is not int
    ):
        raise TypeError("physical state plan has invalid field types")
    return StatePhysicalPlan(
        plan_epoch=plan_epoch,
        request_digest=request_digest,
        representation_signature=representation_signature,
        representations=tuple(
            _provider_from_boundary(plan, f"representations[{position}]")
            for position, plan in enumerate(
                cast(list[object], representations)
            )
        ),
        reserved_bytes=reserved_bytes,
        schema_version=schema_version,
    )


def lower_state_physical_plan(
    request: StateRepresentationRequest,
    decision: StateRepresentationDecision,
    *,
    plan_epoch: int,
) -> StatePhysicalPlan:
    return _physical_from_boundary(
        _core.lower_canonical_state_physical_plan(
            _request_json(request),
            _decision_json(decision),
            plan_epoch,
        )
    )


def validate_state_physical_plan(
    request: StateRepresentationRequest,
    decision: StateRepresentationDecision,
    physical: StatePhysicalPlan,
) -> None:
    _core.validate_canonical_state_physical_plan(
        _request_json(request),
        _decision_json(decision),
        _physical_json(physical),
    )


__all__ = [
    "compile_state_representation",
    "lower_state_physical_plan",
    "validate_state_physical_plan",
    "validate_state_representation",
    "validate_state_representation_reconstruction",
]
