"""Optimizer and state-field authoring values (pllm.SGD, pllm.Field) and the layouts they declare."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
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
from pllm.execution.parameter_projection import validate_custom_projection_operator


@dataclass(frozen=True)
class SGD:
    lr: float

    def __post_init__(self) -> None:
        if not isfinite(self.lr) or self.lr <= 0:
            raise ValueError("SGD requires a positive finite learning rate")


@dataclass(frozen=True)
class Field:
    """A state field with an explicit dtype, or a shape computed from a projection's (in_features, out_features);
    a field read by an adapter gets one copy per target projection. Without a dtype it takes its state's dtype=;
    with neither, storage follows the declared role: accumulators use FP32, other fields use model precision."""

    shape: tuple[int, ...] | Callable[[int, int], tuple[int, ...]]
    dtype: torch.dtype | Literal["auto"] = "auto"

    def size(self, in_features: int, out_features: int) -> tuple[int, ...]:
        """The shape for one target projection's widths, or for the hidden width outside adapters."""
        return self.shape(in_features, out_features) if callable(self.shape) else self.shape

    def at_projection(self, projection: ProjectionShape) -> Field:
        """Bind a field before computing its global and rank-local storage."""
        return self


@dataclass(frozen=True)
class TargetRows(Field):
    """Pending target rows append their projection's target width at layout binding."""

    shape: tuple[int, ...]

    def at_projection(self, projection: ProjectionShape) -> Field:
        """Keep target rows replicated when column output rows are partitioned."""
        return Field((*self.shape, projection.target_size), self.dtype)


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
    applied = effect.applied_fields
    custom = effect.operator is ParameterOverlayOperator.CUSTOM
    expected_arity = (
        len(applied)
        if custom and len(applied) >= 1
        else 2
        if effect.operator is ParameterOverlayOperator.LOW_RANK_ADD
        else 1
    )
    if (
        not effect.targets
        or tuple(sorted(set(effect.targets))) != effect.targets
        or not applied
        or len(applied) != expected_arity
        or len(set(applied)) != expected_arity
        or any(key not in fields for key in applied)
        or not isfinite(effect.scale)
        or (custom and effect.custom_operator is None)
    ):
        raise ValueError("an adapter needs sorted unique targets, the fields its operator applies and a finite scale")
    projections = dimensions.projections(effect.targets)
    leaders = _validate_shared_fields(effect, fields, projections)
    validated_custom: set[tuple[int, int, tuple[tuple[tuple[int, ...], TensorDType], ...]]] = set()
    for index, projection in enumerate(projections):
        target_dtypes = {
            key: dtype(f"{key}_{leaders.get((key, index), index)}", value) for key, value in fields.items()
        }
        if custom:
            assert effect.custom_operator is not None
            inputs, outputs = projection.global_shape
            weight_specs = tuple((fields[key].size(inputs, outputs), target_dtypes[key]) for key in applied)
            sig = (inputs, outputs, weight_specs)
            if sig not in validated_custom:
                validate_custom_projection_operator(
                    effect.custom_operator,
                    inputs,
                    outputs,
                    dimensions.dtype,
                    weight_specs,
                    f"{projection.target.name} of layer {projection.target.layer}",
                )
                validated_custom.add(sig)
        else:
            _check_target_fields(
                effect,
                fields,
                f"{projection.target.name} of layer {projection.target.layer}",
                *projection.global_shape,
                dimensions.dtype,
                target_dtypes,
            )
    if dimensions.tensor_parallel_size > 1:
        if effect.operator is ParameterOverlayOperator.HEAD_ADD:
            raise ValueError("per_head adapters need tensor_parallel_size=1; use dense or low_rank deltas under TP")
        if custom:
            raise ValueError(
                "custom projection operators need tensor_parallel_size=1; use dense or low_rank deltas under TP"
            )
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
        if (key, index) not in leaders
    )


def _validate_shared_fields(
    effect: ProjectionDelta,
    fields: Mapping[str, Field],
    projections: tuple[ProjectionShape, ...],
) -> dict[tuple[str, int], int]:
    """Validate per-field layer sharing groups and return (field, follower_index) -> leader_index."""
    if not effect.share:
        return {}
    if effect.route is not None:
        raise ValueError("Adapter(share=...) cannot be combined with LayerRoute.adapter")
    target_index = {projection.target: i for i, projection in enumerate(projections)}
    leaders: dict[tuple[str, int], int] = {}
    for field_name, groups in effect.share:
        if field_name not in fields:
            raise ValueError(f"Adapter(share=...): {field_name!r} is not a field of state {effect.state!r}")
        for group in groups:
            leader_layer = group[0]
            for follower_layer in group[1:]:
                for i, projection in enumerate(projections):
                    if projection.target.layer != follower_layer:
                        continue
                    leader_target = replace(projection.target, layer=leader_layer)
                    if leader_target not in target_index:
                        raise ValueError(
                            f"Adapter(share[{field_name!r}]): follower target {projection.target} has no matching "
                            f"leader target {leader_target} in Adapter.targets"
                        )
                    leader_idx = target_index[leader_target]
                    leader_proj = projections[leader_idx]
                    if (
                        projection.input_size,
                        projection.output_size,
                        projection.input_partitions,
                        projection.output_partitions,
                    ) != (
                        leader_proj.input_size,
                        leader_proj.output_size,
                        leader_proj.input_partitions,
                        leader_proj.output_partitions,
                    ):
                        raise ValueError(
                            f"Adapter(share[{field_name!r}]): {projection.target} and {leader_target} have different "
                            "projection shapes and cannot share a field"
                        )
                    leaders[field_name, i] = leader_idx
    return leaders


def routed_overlay_fields(
    fields: Mapping[str, Field],
    effect: ProjectionDelta,
    dimensions: Dimensions,
    dtype: Callable[[str, Field], TensorDType],
    *,
    layer_bank: bool,
) -> tuple[LayoutField, ...]:
    """Validate projection homogeneity across `effect.route.over` and allocate single-slot or per-layer fields."""
    assert effect.route is not None
    projections = dimensions.projections(effect.targets)
    by_name: dict[str, list[ProjectionShape]] = {}
    for projection in projections:
        by_name.setdefault(projection.target.name, []).append(projection)
    for name, group in by_name.items():
        first = group[0]
        if any(
            (item.input_size, item.output_size, item.input_partitions, item.output_partitions)
            != (first.input_size, first.output_size, first.input_partitions, first.output_partitions)
            for item in group[1:]
        ):
            raise ValueError(
                f"LayerRoute.adapter: {name} does not have identical shapes across over={effect.route.over}"
            )
    if layer_bank:
        if len(effect.target_names) == 1:

            def bank_dtype(key_arg: str, value: Field) -> TensorDType:
                if key_arg in fields:
                    return dtype(f"{key_arg}_{projections[0].target.layer}", value)
                head, _, tail = key_arg.rpartition("_")
                return dtype(f"{head}_{projections[int(tail)].target.layer}", value)

            validated = overlay_fields(fields, effect, dimensions, bank_dtype)
            return tuple(
                replace(item, name=f"{key}_{projection.target.layer}")
                for item, (projection, key) in zip(
                    validated,
                    ((projection, key) for projection in projections for key in fields),
                    strict=True,
                )
            )
        return overlay_fields(fields, effect, dimensions, dtype)
    # Single-slot O(1) mode: allocate one set of parameter fields per target projection name (using the first layer).
    representatives = replace(effect, targets=tuple(by_name[name][0].target for name in effect.target_names))
    return overlay_fields(fields, representatives, dimensions, dtype)


def overlay_field(name: str, field: Field, projection: ProjectionShape, size: int, dtype: TensorDType) -> LayoutField:
    """Partition the projection's feature axis; constant-sized fields remain replicated."""
    field = field.at_projection(projection)
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
        case ParameterOverlayOperator.CUSTOM:
            return
    if not valid:
        raise ValueError(
            f"fields {keys} must be shaped {rule} for the {projection} (in_features={inputs}, out_features={outputs}); "
            f"got {shapes}"
        )
    allowed = (model, TensorDType.FLOAT32, TensorDType.FLOAT64)
    for key in keys:
        if dtypes[key] not in allowed:
            raise ValueError(f"field {key!r} must hold one of {allowed} for {effect.operator}; set dtype=torch.float32")
