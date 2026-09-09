"""Closed Python–Rust boundary for frozen tensor-layout interning."""

from __future__ import annotations


from pllm import _core
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.contracts.tensor_layout import (
    TensorLayoutDecision,
    TensorLayoutRequest,
)


def intern_tensor_layouts(request: TensorLayoutRequest) -> TensorLayoutDecision:
    return decode_boundary(_core.intern_tensor_layouts(encode_boundary(request)), TensorLayoutDecision)


def validate_tensor_layouts(
    request: TensorLayoutRequest,
    decision: TensorLayoutDecision,
) -> None:
    _core.validate_tensor_layouts(
        encode_boundary(request),
        encode_boundary(decision),
    )


__all__ = [
    "intern_tensor_layouts",
    "validate_tensor_layouts",
]
