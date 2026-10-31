"""Frozen tensor-layout vocabulary for canonical PTTS interning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

import torch

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

    @property
    def torch(self) -> torch.dtype:
        """The native torch dtype."""
        return _TORCH_DTYPES[self]

    @property
    def itemsize(self) -> int:
        """Bytes per element."""
        return torch.empty(0, dtype=self.torch).element_size()


_TORCH_DTYPES = {
    TensorDType.BOOL: torch.bool,
    TensorDType.UINT8: torch.uint8,
    TensorDType.INT32: torch.int32,
    TensorDType.INT64: torch.int64,
    TensorDType.FLOAT16: torch.float16,
    TensorDType.BFLOAT16: torch.bfloat16,
    TensorDType.FLOAT32: torch.float32,
    TensorDType.FLOAT64: torch.float64,
}


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


@dataclass(frozen=True, slots=True)
class TensorLayoutDecision:
    bindings: tuple[TensorLayoutBinding, ...]
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
