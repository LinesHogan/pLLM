"""Parameter effects and model advancement declarations shared by authoring and backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import prod

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
    layer: int
    module: ParameterModuleKind
    role: ParameterRole

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("parameter target layer must be resolved and non-negative")


@dataclass(frozen=True, slots=True)
class ParameterTargetShape:
    target: ParameterTarget
    input_size: int
    output_size: int

    def __post_init__(self) -> None:
        if self.input_size < 1 or self.output_size < 1:
            raise ValueError("parameter target dimensions must be positive")


@dataclass(frozen=True, slots=True)
class FieldView:
    """A logical tensor inside one PTTS field, shared by producers and model readers."""

    field: LayoutFieldId
    offset: int
    shape: tuple[int, ...]

    def read(self, tensor: torch.Tensor, *, pooled: bool = False) -> torch.Tensor:
        """Borrow the named slice while preserving a pool's leading slot axis."""
        prefix = (len(tensor),) if pooled else ()
        return (
            tensor.reshape(*prefix, -1).narrow(int(pooled), self.offset, prod(self.shape)).view((*prefix, *self.shape))
        )


@dataclass(frozen=True, slots=True)
class ParameterOverlayFieldBinding:
    target: ParameterTarget
    operator: ParameterOverlayOperator
    fields: tuple[FieldView, ...]
    resource: str
    scale: float = 1.0


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


__all__ = [
    "ParameterEffect",
    "ParameterModuleKind",
    "ParameterOverlayFieldBinding",
    "ParameterOverlayOperator",
    "ParameterRole",
    "ParameterTarget",
    "CapturedParameterOperand",
]
