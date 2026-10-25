"""Parameter effects and model advancement declarations shared by authoring and backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

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
class ParameterOverlayFieldBinding:
    target: ParameterTarget
    operator: ParameterOverlayOperator
    fields: tuple[LayoutFieldId, ...]
    resource: str
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class CapturedParameterOperand:
    identity: RequestIdentity
    target: ParameterTarget
    phase: HiddenPhase
    hidden: torch.Tensor
    desired: torch.Tensor
    base_weight: torch.Tensor


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
