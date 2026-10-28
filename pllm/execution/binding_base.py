"""Optimizer and buffer authoring values used by workload markers and binding functions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite

import torch

from pllm.contracts.parameter_overlay import (
    ParameterEffect,
    ParameterOverlayOperator,
)
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.declaration import Dimensions, Field


@dataclass(frozen=True)
class SGD:
    lr: float

    def __post_init__(self) -> None:
        if not isfinite(self.lr) or self.lr <= 0:
            raise ValueError("SGD requires a positive finite learning rate")


@dataclass(frozen=True)
class Buffer:
    shape: tuple[int, ...] | Callable[[int, int], tuple[int, ...]]
    dtype: torch.dtype | None = None


def buffer_field(name: str, buffer: Buffer, dimensions: Callable[[Dimensions], tuple[int, int]]) -> Field:
    shape = buffer.shape

    def size(model: Dimensions) -> tuple[int, ...]:
        return shape(*dimensions(model)) if callable(shape) else shape

    return Field(
        name,
        size,
        (lambda model: model.dtype) if buffer.dtype is None else TensorDType(str(buffer.dtype).removeprefix("torch.")),
        8 if buffer.dtype is torch.int64 else 16,
    )


def overlay_fields(buffers: Mapping[str, Buffer], effect: ParameterEffect) -> tuple[Field, ...]:
    """One copy of the buffers per overlay target, laid out target by target and shaped by that target."""
    arity = 2 if effect.operator is ParameterOverlayOperator.LOW_RANK_ADD else 1
    applied = effect.applied_fields
    if (
        not effect.targets
        or tuple(sorted(set(effect.targets))) != effect.targets
        or len(applied) != arity
        or len(set(applied)) != arity
        or any(key not in buffers for key in applied)
        or not isfinite(effect.scale)
    ):
        raise ValueError("overlay requires ordered unique targets, exact operator fields and a finite scale")
    return tuple(
        buffer_field(f"{key}_{index}", buffer, lambda m, i=index: m.target_shapes[i])
        for index in range(len(effect.targets))
        for key, buffer in buffers.items()
    )
