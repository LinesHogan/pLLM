"""Optimizer and state-field authoring values (pllm.SGD, pllm.Field) and the layouts they declare."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch

from pllm.contracts.parameter_overlay import (
    ParameterOverlayOperator,
    ProjectionDelta,
    ProjectionShape,
)
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.declaration import Dimensions, FieldShard, LayoutField


@dataclass(frozen=True)
class SGD:
    lr: float

    def __post_init__(self) -> None:
        if not isfinite(self.lr) or self.lr <= 0:
            raise ValueError("SGD requires a positive finite learning rate")


@dataclass(frozen=True)
class Field:
    """A state field with an explicit dtype, or a shape computed from a projection's (in_features, out_features);
    a field read by an adapter gets one copy per target projection. Storage follows its declared role by default:
    accumulators use FP32, other fields use model precision. An explicit dtype overrides this; None follows the model."""

    shape: tuple[int, ...] | Callable[[int, int], tuple[int, ...]]
    dtype: torch.dtype | Literal["auto"] | None = "auto"

    def size(self, in_features: int, out_features: int) -> tuple[int, ...]:
        """The shape for one target projection's widths, or for the hidden width outside adapters."""
        return self.shape(in_features, out_features) if callable(self.shape) else self.shape


def layout_field(
    name: str, field: Field, dimensions: Callable[[Dimensions], tuple[int, int]], dtype: TensorDType
) -> LayoutField:
    """Resolve a projection-sized field with the storage dtype chosen by authoring."""
    return LayoutField(
        name, lambda model: field.size(*dimensions(model)), dtype, 8 if dtype is TensorDType.INT64 else 16
    )


def overlay_fields(
    fields: Mapping[str, Field],
    effect: ProjectionDelta,
    dimensions: Dimensions,
    dtype: Callable[[str, Field], TensorDType],
) -> tuple[LayoutField, ...]:
    """One copy of the fields per overlay target, laid out target by target and shaped by that target."""
    arity = 2 if effect.operator is ParameterOverlayOperator.LOW_RANK_ADD else 1
    applied = effect.applied_fields
    if (
        not effect.targets
        or tuple(sorted(set(effect.targets))) != effect.targets
        or len(applied) != arity
        or len(set(applied)) != arity
        or any(key not in fields for key in applied)
        or not isfinite(effect.scale)
    ):
        raise ValueError("an adapter needs sorted unique targets, the fields its operator applies and a finite scale")
    projections = dimensions.projections(effect.targets)
    for index, projection in enumerate(projections):
        _check_target_fields(
            effect,
            fields,
            f"{projection.target.name} of layer {projection.target.layer}",
            *projection.global_shape,
            dimensions.dtype,
            {key: dtype(f"{key}_{index}", value) for key, value in fields.items()},
        )
    if dimensions.tensor_parallel_size > 1:
        if effect.operator is ParameterOverlayOperator.HEAD_ADD:
            raise ValueError("per_head adapters need tensor_parallel_size=1; use dense or low_rank deltas under TP")
        for projection in projections:
            _check_target_fields(
                effect,
                fields,
                "rank-local projection",
                projection.input_size,
                projection.output_size,
                dimensions.dtype,
                {key: dtype(key, value) for key, value in fields.items()},
            )
    return tuple(
        overlay_field(
            f"{key}_{index}", field, projection, dimensions.tensor_parallel_size, dtype(f"{key}_{index}", field)
        )
        for index, projection in enumerate(projections)
        for key, field in fields.items()
    )


def overlay_field(name: str, field: Field, projection: ProjectionShape, size: int, dtype: TensorDType) -> LayoutField:
    """Partition the projection's feature axis; constant-sized fields remain replicated."""
    shape = field.size(*projection.global_shape)
    local = field.size(projection.input_size, projection.output_size)
    axes = tuple(i for i, (whole, part) in enumerate(zip(shape, local, strict=True)) if whole != part)
    shard = None
    if axes:
        if len(axes) != 1 or shape[axes[0]] != local[axes[0]] * projection.partitions:
            raise ValueError(f"{name}: TP fields must partition one feature axis evenly; use tensor_parallel_size=1")
        shard = FieldShard(shape, axes[0], size // projection.partitions)
    return LayoutField(name, local, dtype, 8 if dtype is TensorDType.INT64 else 16, shard)


def _check_target_fields(
    effect: ProjectionDelta,
    fields: Mapping[str, Field],
    projection: str,
    inputs: int,
    outputs: int,
    model: TensorDType,
    dtypes: Mapping[str, TensorDType],
) -> None:
    """Reject the fields one target projection cannot read, naming the field to fix."""
    keys = effect.applied_fields
    shapes = tuple(fields[key].size(inputs, outputs) for key in keys)
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
            f"fields {keys} must be shaped {rule} for the {projection} (in_features={inputs}, out_features={outputs}); "
            f"got {shapes}"
        )
    allowed = (model, TensorDType.FLOAT32, TensorDType.FLOAT64)
    for key in keys:
        if dtypes[key] not in allowed:
            raise ValueError(f"field {key!r} must hold one of {allowed} for {effect.operator}; set dtype=torch.float32")
