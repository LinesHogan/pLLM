"""Frozen tensor-layout vocabulary for canonical PTTS interning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType

from tllm.contracts.state_ref import LayoutId


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


class MutationGranularity(StrEnum):
    WHOLE_TENSOR = "whole_tensor"
    ROW = "row"
    PAGE = "page"


@dataclass(frozen=True, slots=True)
class TensorFieldSpec:
    field_id: LayoutFieldId
    shape: tuple[int, ...]
    dtype: TensorDType
    alignment_bytes: int
    mutation_granularity: MutationGranularity

    def __post_init__(self) -> None:
        if int(self.field_id) < 0 or any(
            dimension < 1 for dimension in self.shape
        ):
            raise ValueError("tensor field identity and shape must be positive")
        if (
            self.alignment_bytes < 1
            or self.alignment_bytes & (self.alignment_bytes - 1)
        ):
            raise ValueError("tensor field alignment must be a power of two")


@dataclass(frozen=True, slots=True)
class TensorLayoutSpec:
    namespace: str
    fields: tuple[TensorFieldSpec, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.namespace or self.schema_version != 1:
            raise ValueError("invalid tensor layout namespace or schema")
        if tuple(int(field.field_id) for field in self.fields) != tuple(
            range(len(self.fields))
        ):
            raise ValueError("tensor layout field IDs must be dense and ordered")


@dataclass(frozen=True, slots=True)
class StateTensorLayout:
    resource: str
    layout: TensorLayoutSpec

    def __post_init__(self) -> None:
        if not self.resource:
            raise ValueError("state tensor layout requires a resource")


@dataclass(frozen=True, slots=True)
class TensorLayoutRequest:
    layouts: tuple[StateTensorLayout, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        resources = tuple(item.resource for item in self.layouts)
        if (
            not resources
            or tuple(sorted(resources)) != resources
            or len(set(resources)) != len(resources)
            or self.schema_version != 1
        ):
            raise ValueError(
                "tensor layout request needs sorted unique resources"
            )


@dataclass(frozen=True, slots=True)
class TensorLayoutBinding:
    resource: str
    layout_id: LayoutId
    layout_namespace: str
    layout_digest: str

    def __post_init__(self) -> None:
        if not self.resource or int(self.layout_id) < 0:
            raise ValueError("invalid tensor layout binding identity")
        for value in (self.layout_namespace, self.layout_digest):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    "tensor layout namespace/digest must be lowercase SHA-256"
                )


@dataclass(frozen=True, slots=True)
class TensorLayoutDecision:
    request_digest: str
    layout_namespace: str
    bindings: tuple[TensorLayoutBinding, ...]
    signature: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for value in (
            self.request_digest,
            self.layout_namespace,
            self.signature,
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("tensor layout decision hashes must be SHA-256")
        resources = tuple(binding.resource for binding in self.bindings)
        if (
            tuple(sorted(resources)) != resources
            or len(set(resources)) != len(resources)
            or any(
                binding.layout_namespace != self.layout_namespace
                for binding in self.bindings
            )
            or self.schema_version != 1
        ):
            raise ValueError("invalid tensor layout decision bindings")


__all__ = [
    "LayoutFieldId",
    "MutationGranularity",
    "StateTensorLayout",
    "TensorDType",
    "TensorFieldSpec",
    "TensorLayoutBinding",
    "TensorLayoutDecision",
    "TensorLayoutRequest",
    "TensorLayoutSpec",
]
