"""Closed Python–Rust boundary for frozen tensor-layout interning."""

from __future__ import annotations

import json
from typing import TypedDict, cast

from tllm import _core
from tllm.contracts.state_ref import LayoutId
from tllm.contracts.tensor_layout import (
    TensorLayoutBinding,
    TensorLayoutDecision,
    TensorLayoutRequest,
)


class _FieldWire(TypedDict):
    field_id: int
    shape: list[int]
    dtype: str
    alignment_bytes: int
    mutation_granularity: str


class _LayoutWire(TypedDict):
    namespace: str
    fields: list[_FieldWire]
    schema_version: int


class _StateLayoutWire(TypedDict):
    resource: str
    layout: _LayoutWire


class _RequestWire(TypedDict):
    layouts: list[_StateLayoutWire]
    schema_version: int


def _request_json(request: TensorLayoutRequest) -> str:
    wire = _RequestWire(
        layouts=[
            _StateLayoutWire(
                resource=item.resource,
                layout=_LayoutWire(
                    namespace=item.layout.namespace,
                    fields=[
                        _FieldWire(
                            field_id=int(field.field_id),
                            shape=list(field.shape),
                            dtype=field.dtype.value,
                            alignment_bytes=field.alignment_bytes,
                            mutation_granularity=(
                                field.mutation_granularity.value
                            ),
                        )
                        for field in item.layout.fields
                    ],
                    schema_version=item.layout.schema_version,
                ),
            )
            for item in request.layouts
        ],
        schema_version=request.schema_version,
    )
    return json.dumps(
        wire,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _decision_json(decision: TensorLayoutDecision) -> str:
    return json.dumps(
        {
            "request_digest": decision.request_digest,
            "layout_namespace": decision.layout_namespace,
            "bindings": [
                {
                    "resource": binding.resource,
                    "layout_id": int(binding.layout_id),
                    "layout_namespace": binding.layout_namespace,
                    "layout_digest": binding.layout_digest,
                }
                for binding in decision.bindings
            ],
            "signature": decision.signature,
            "schema_version": decision.schema_version,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _decision_from_boundary(payload: str) -> TensorLayoutDecision:
    decoded = cast(object, json.loads(payload))
    if type(decoded) is not dict:
        raise TypeError("tensor layout decision must be an object")
    fields = cast(dict[object, object], decoded)
    if set(fields) != {
        "request_digest",
        "layout_namespace",
        "bindings",
        "signature",
        "schema_version",
    }:
        raise ValueError("tensor layout decision has the wrong fields")
    request_digest = fields["request_digest"]
    layout_namespace = fields["layout_namespace"]
    bindings = fields["bindings"]
    signature = fields["signature"]
    schema_version = fields["schema_version"]
    if (
        type(request_digest) is not str
        or type(layout_namespace) is not str
        or type(bindings) is not list
        or type(signature) is not str
        or type(schema_version) is not int
    ):
        raise TypeError("tensor layout decision has invalid field types")
    parsed: list[TensorLayoutBinding] = []
    for position, binding in enumerate(cast(list[object], bindings)):
        if type(binding) is not dict:
            raise TypeError(f"bindings[{position}] must be an object")
        binding_fields = cast(dict[object, object], binding)
        if set(binding_fields) != {
            "resource",
            "layout_id",
            "layout_namespace",
            "layout_digest",
        }:
            raise ValueError(f"bindings[{position}] has the wrong fields")
        resource = binding_fields["resource"]
        layout_id = binding_fields["layout_id"]
        binding_namespace = binding_fields["layout_namespace"]
        layout_digest = binding_fields["layout_digest"]
        if (
            type(resource) is not str
            or type(layout_id) is not int
            or type(binding_namespace) is not str
            or type(layout_digest) is not str
        ):
            raise TypeError(f"bindings[{position}] has invalid field types")
        parsed.append(
            TensorLayoutBinding(
                resource=resource,
                layout_id=LayoutId(layout_id),
                layout_namespace=binding_namespace,
                layout_digest=layout_digest,
            )
        )
    return TensorLayoutDecision(
        request_digest=request_digest,
        layout_namespace=layout_namespace,
        bindings=tuple(parsed),
        signature=signature,
        schema_version=schema_version,
    )


def intern_tensor_layouts(request: TensorLayoutRequest) -> TensorLayoutDecision:
    return _decision_from_boundary(
        _core.intern_canonical_tensor_layouts(_request_json(request))
    )


def validate_tensor_layouts(
    request: TensorLayoutRequest,
    decision: TensorLayoutDecision,
) -> None:
    _core.validate_canonical_tensor_layouts(
        _request_json(request),
        _decision_json(decision),
    )


def encode_tensor_layout_request(request: TensorLayoutRequest) -> str:
    return _request_json(request)


def encode_tensor_layout_decision(decision: TensorLayoutDecision) -> str:
    return _decision_json(decision)


__all__ = [
    "encode_tensor_layout_decision",
    "encode_tensor_layout_request",
    "intern_tensor_layouts",
    "validate_tensor_layouts",
]
