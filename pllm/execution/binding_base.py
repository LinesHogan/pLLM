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

    def size(self, inputs: int, outputs: int) -> tuple[int, ...]:
        """The shape for one overlay target's widths, or for the hidden width outside overlays."""
        return self.shape(inputs, outputs) if callable(self.shape) else self.shape

    @property
    def tensor_dtype(self) -> TensorDType | None:
        """The declared dtype; None follows the model."""
        return None if self.dtype is None else TensorDType(str(self.dtype).removeprefix("torch."))


def buffer_field(name: str, buffer: Buffer, dimensions: Callable[[Dimensions], tuple[int, int]]) -> Field:
    dtype = buffer.tensor_dtype
    return Field(
        name,
        lambda model: buffer.size(*dimensions(model)),
        (lambda model: model.dtype) if dtype is None else dtype,
        8 if buffer.dtype is torch.int64 else 16,
    )


def overlay_fields(buffers: Mapping[str, Buffer], effect: ParameterEffect, dimensions: Dimensions) -> tuple[Field, ...]:
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
    for target, (inputs, outputs) in zip(effect.targets, dimensions.target_shapes, strict=True):
        _check_target_buffers(
            effect, buffers, f"{target.role} of layer {target.layer}", inputs, outputs, dimensions.dtype
        )
    return tuple(
        buffer_field(f"{key}_{index}", buffer, lambda m, i=index: m.target_shapes[i])
        for index in range(len(effect.targets))
        for key, buffer in buffers.items()
    )


def _check_target_buffers(
    effect: ParameterEffect,
    buffers: Mapping[str, Buffer],
    projection: str,
    inputs: int,
    outputs: int,
    model: TensorDType,
) -> None:
    """Reject the Buffers one target projection cannot read, naming the Buffer to fix."""
    keys = effect.applied_fields
    shapes = tuple(buffers[key].size(inputs, outputs) for key in keys)
    match effect.operator:
        case ParameterOverlayOperator.DENSE_ADD:
            rule, valid = "(outputs, inputs)", shapes == ((outputs, inputs),)
        case ParameterOverlayOperator.LOW_RANK_ADD:
            rank = shapes[0][0] if shapes[0] else 0
            rule, valid = "(rank, inputs) and (outputs, rank)", rank > 0 and shapes == ((rank, inputs), (outputs, rank))
        case ParameterOverlayOperator.HEAD_ADD:
            (shape,) = shapes
            rule = "(heads, width, width) with heads * width == inputs == outputs"
            valid = (
                len(shape) == 3 and shape[0] > 0 and shape[1] == shape[2] and shape[0] * shape[1] == inputs == outputs
            )
    if not valid:
        raise ValueError(
            f"overlay Buffers {keys} must be shaped {rule} for the {projection} (inputs={inputs}, outputs={outputs}); "
            f"got {shapes}"
        )
    required = TensorDType.FLOAT32 if effect.operator is ParameterOverlayOperator.HEAD_ADD else model
    for key in keys:
        if (buffers[key].tensor_dtype or model) is not required:
            raise ValueError(
                f"overlay Buffer {key!r} must hold {required} for {effect.operator}; set dtype=torch.{required}"
            )
