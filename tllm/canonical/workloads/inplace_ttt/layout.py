"""Frozen PTTS field layout for multi-target dense fast weights."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTConfig,
    InPlaceTTTSpecialization,
)
from tllm.contracts.parameter_overlay import (
    ParameterOverlayFieldBinding,
    ParameterOverlayOperator,
    ParameterTarget,
)
from tllm.contracts.tensor_layout import (
    LayoutFieldId,
    MutationGranularity,
    TensorDType,
    TensorFieldSpec,
    TensorLayoutSpec,
)


@dataclass(frozen=True, slots=True)
class DenseFastWeightFieldBinding:
    target: ParameterTarget
    delta_weight: LayoutFieldId
    tail_input: LayoutFieldId
    tail_target: LayoutFieldId
    tail_length: LayoutFieldId


@dataclass(frozen=True, slots=True)
class InPlaceTTTStateLayout:
    tensor_layout: TensorLayoutSpec
    targets: tuple[DenseFastWeightFieldBinding, ...]

    @property
    def overlay_bindings(self) -> tuple[ParameterOverlayFieldBinding, ...]:
        return tuple(
            ParameterOverlayFieldBinding(
                target.target,
                ParameterOverlayOperator.DENSE_ADD,
                (target.delta_weight,),
            )
            for target in self.targets
        )

    def fields_for(self, target: ParameterTarget) -> DenseFastWeightFieldBinding:
        for binding in self.targets:
            if binding.target == target:
                return binding
        raise ValueError("dense fast-weight target is outside the frozen layout")


def build_inplace_ttt_state_layout(
    config: InPlaceTTTConfig,
    specialization: InPlaceTTTSpecialization,
) -> InPlaceTTTStateLayout:
    if tuple(target.target for target in specialization.targets) != config.targets:
        raise ValueError("In-Place-TTT layout targets differ from specialization")
    fields: list[TensorFieldSpec] = []
    bindings: list[DenseFastWeightFieldBinding] = []
    tail_capacity = max(1, config.chunk_size - 1)
    for target in specialization.targets:
        first_field = len(fields)
        bindings.append(
            DenseFastWeightFieldBinding(
                target.target,
                LayoutFieldId(first_field),
                LayoutFieldId(first_field + 1),
                LayoutFieldId(first_field + 2),
                LayoutFieldId(first_field + 3),
            )
        )
        fields.extend(
            (
                TensorFieldSpec(
                    LayoutFieldId(first_field),
                    (target.output_size, target.input_size),
                    specialization.tensor_dtype,
                    16,
                    MutationGranularity.WHOLE_TENSOR,
                ),
                TensorFieldSpec(
                    LayoutFieldId(first_field + 1),
                    (tail_capacity, target.input_size),
                    specialization.tensor_dtype,
                    16,
                    MutationGranularity.WHOLE_TENSOR,
                ),
                TensorFieldSpec(
                    LayoutFieldId(first_field + 2),
                    (tail_capacity, target.output_size),
                    specialization.tensor_dtype,
                    16,
                    MutationGranularity.WHOLE_TENSOR,
                ),
                TensorFieldSpec(
                    LayoutFieldId(first_field + 3),
                    (),
                    TensorDType.INT64,
                    8,
                    MutationGranularity.WHOLE_TENSOR,
                ),
            )
        )
    return InPlaceTTTStateLayout(
        TensorLayoutSpec(
            "canonical.dense_fast_weight.state@1",
            tuple(fields),
        ),
        tuple(bindings),
    )


__all__ = [
    "DenseFastWeightFieldBinding",
    "InPlaceTTTStateLayout",
    "build_inplace_ttt_state_layout",
]
