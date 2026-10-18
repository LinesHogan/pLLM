"""Frozen tensor-layout vocabulary for canonical PTTS interning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

from pllm.contracts.state_ref import LayoutId

LayoutFieldId = NewType("LayoutFieldId", int)


class TensorDType(StrEnum):
    BOOL = "bool"
    UINT8 = "uint8"
    INT32 = "int32"
    INT64 = "int64"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    field_id: LayoutFieldId
    shape: tuple[int, ...]
    dtype: TensorDType
    alignment_bytes: int


@dataclass(frozen=True, slots=True)
class TensorLayoutSpec:
    namespace: str
    fields: tuple[TensorFieldSpec, ...]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class StateTensorLayout:
    resource: str
    layout: TensorLayoutSpec


@dataclass(frozen=True, slots=True)
class TensorLayoutRequest:
    layouts: tuple[StateTensorLayout, ...]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class TensorLayoutBinding:
    resource: str
    layout_id: LayoutId
    layout_namespace: str
    layout_digest: str


@dataclass(frozen=True, slots=True)
class TensorLayoutDecision:
    request_digest: str
    layout_namespace: str
    bindings: tuple[TensorLayoutBinding, ...]
    signature: str
    schema_version: int = 1


__all__ = [
    "LayoutFieldId",
    "StateTensorLayout",
    "TensorDType",
    "TensorFieldSpec",
    "TensorLayoutBinding",
    "TensorLayoutDecision",
    "TensorLayoutRequest",
    "TensorLayoutSpec",
]
