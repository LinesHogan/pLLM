"""Closed logical bindings for immutable parameter-overlay publications."""

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
    scale: float = 1.0


@dataclass(frozen=True, slots=True)
class CapturedParameterOperand:
    identity: RequestIdentity
    target: ParameterTarget
    phase: HiddenPhase
    hidden: torch.Tensor
    desired: torch.Tensor
    base_weight: torch.Tensor


__all__ = [
    "ParameterModuleKind",
    "ParameterOverlayFieldBinding",
    "ParameterOverlayOperator",
    "ParameterRole",
    "ParameterTarget",
    "CapturedParameterOperand",
]
