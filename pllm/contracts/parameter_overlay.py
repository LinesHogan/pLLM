"""Parameter effects and model advancement declarations shared by authoring and backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from math import prod
from typing import Self

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.tensor_layout import LayoutFieldId


class ParameterModuleKind(StrEnum):
    ATTENTION = "attention"
    MLP = "mlp"


class ParameterRole(StrEnum):
    DOWN_PROJECTION_WEIGHT = "down_projection_weight"
    QUERY_PROJECTION_WEIGHT = "query_projection_weight"
    VALUE_PROJECTION_WEIGHT = "value_projection_weight"


class ParameterOverlayOperator(StrEnum):
    DENSE_ADD = "dense_add"
    LOW_RANK_ADD = "low_rank_add"
    HEAD_ADD = "head_add"


@dataclass(frozen=True, slots=True, order=True)
class ParameterTarget:
    layer: int | None
    module: ParameterModuleKind
    role: ParameterRole

    def __post_init__(self) -> None:
        if self.layer is not None and self.layer < 0:
            raise ValueError("parameter target layer must be resolved and non-negative")


def expand_targets(targets: tuple[ParameterTarget, ...], layers: int) -> tuple[ParameterTarget, ...]:
    """Resolve layer=None declarations against the loaded model, in projection order."""
    if all(target.layer is not None for target in targets):
        return targets
    return tuple(
        sorted(
            ParameterTarget(layer, target.module, target.role)
            for target in targets
            for layer in (range(layers) if target.layer is None else (target.layer,))
        )
    )


@dataclass(frozen=True, slots=True)
class ParameterTargetShape:
    target: ParameterTarget
    input_size: int
    output_size: int


@dataclass(frozen=True, slots=True)
class FieldView:
    """A logical tensor inside one PTTS field, shared by producers and model readers."""

    field: LayoutFieldId
    offset: int
    shape: tuple[int, ...]

    def read(self, tensor: torch.Tensor, *, pooled: bool = False) -> torch.Tensor:
        """Borrow the named slice while preserving a pool's leading slot axis."""
        prefix = (len(tensor),) if pooled else ()
        flat = tensor.reshape(*prefix, -1)
        # Even a full-width narrow allocates and copies a full gradient in backward.
        if self.offset or prod(self.shape) != flat.shape[-1]:
            flat = flat.narrow(int(pooled), self.offset, prod(self.shape))
        return flat.view((*prefix, *self.shape))


@dataclass(frozen=True, slots=True)
class ParameterOverlayFieldBinding:
    target: ParameterTarget
    operator: ParameterOverlayOperator
    fields: tuple[FieldView, ...]
    resource: str
    scale: float = 1.0
    # The overlay function reads operand("weight"): only then does a quantized projection get a dense copy.
    reads_weight: bool = False


@dataclass(frozen=True, slots=True)
class CapturedParameterOperand:
    """Projection tensors and a scalar prompt-end fact, live through captured replay."""

    identity: RequestIdentity
    target: ParameterTarget
    phase: HiddenPhase
    hidden: torch.Tensor
    desired: torch.Tensor
    base_weight: torch.Tensor
    prompt_end: torch.Tensor


@dataclass(frozen=True)
class ParameterEffect:
    """Apply request state at original model projections."""

    state: str
    targets: tuple[ParameterTarget, ...]
    weight: str = "value"
    operator: ParameterOverlayOperator = ParameterOverlayOperator.DENSE_ADD
    factors: tuple[str, str] = ("a", "b")
    scale: float = 1.0

    @property
    def applied_fields(self) -> tuple[str, ...]:
        """The fields consumed by the declared projection operator."""
        return self.factors if self.operator is ParameterOverlayOperator.LOW_RANK_ADD else (self.weight,)

    def expand(self, layers: int) -> Self:
        """The same effect with layer=None targets resolved against a model of `layers` layers."""
        return replace(self, targets=expand_targets(self.targets, layers))

    def field_bindings(
        self, view: Callable[[str], FieldView], *, reads_weight: bool = False
    ) -> tuple[ParameterOverlayFieldBinding, ...]:
        """One binding per target; `view` finds each applied field's copy for target i, named `{field}_{i}`."""
        return tuple(
            ParameterOverlayFieldBinding(
                target,
                self.operator,
                tuple(view(f"{key}_{i}") for key in self.applied_fields),
                self.state,
                self.scale,
                reads_weight,
            )
            for i, target in enumerate(self.targets)
        )


__all__ = [
    "ParameterEffect",
    "ParameterModuleKind",
    "ParameterOverlayFieldBinding",
    "ParameterOverlayOperator",
    "ParameterRole",
    "ParameterTarget",
    "CapturedParameterOperand",
]
