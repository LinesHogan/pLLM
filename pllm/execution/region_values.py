"""Immutable mathematical values, independent of resource and completion owners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

import torch

from pllm.contracts.region_program import BankType, DataType, ScalarType, TensorType, resolve_extent


@dataclass(frozen=True, slots=True)
class Bank:
    elements: tuple[Data, ...]


Data: TypeAlias = int | float | bool | torch.Tensor | Bank


def copy_data(value: Data) -> Data:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Bank):
        return Bank(tuple(copy_data(x) for x in value.elements))
    return value


def check_data(value: Data, type_: DataType, sizes: Mapping[str, int] = {}) -> None:
    if isinstance(type_, TensorType):
        if not isinstance(value, torch.Tensor) or (
            tuple(value.shape),
            str(value.dtype),
            type_.device if value.is_meta else str(value.device),
        ) != (
            type_.shape,
            type_.dtype,
            type_.device,
        ):
            raise ValueError("region tensor violates shape, dtype or device guard")
    elif isinstance(type_, BankType):
        if not isinstance(value, Bank) or len(value.elements) != resolve_extent(type_.size, sizes):
            raise ValueError("region bank violates its logical extent")
        for item in value.elements:
            check_data(item, type_.element, sizes)
    elif type_ is ScalarType.BOOL:
        if type(value) is not bool:
            raise ValueError("region boolean type mismatch")
    elif type_ is ScalarType.FLOAT:
        if type(value) is not float:
            raise ValueError("region floating scalar type mismatch")
    elif type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError("region integer/index must fit signed 64-bit range")
