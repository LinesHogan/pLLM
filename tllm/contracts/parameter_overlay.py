"""Closed logical bindings for immutable parameter-overlay publications."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.state_ref import StateRef
from tllm.contracts.tensor_layout import LayoutFieldId


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

    def __post_init__(self) -> None:
        if not self.fields or len(self.fields) != len(set(self.fields)):
            raise ValueError("parameter overlay requires unique state fields")


@dataclass(frozen=True, slots=True)
class ParameterOverlayPublication:
    """One atomic StateRef publication exposed to a backend adapter."""

    identity: RequestIdentity
    ref: StateRef
    bindings: tuple[ParameterOverlayFieldBinding, ...]

    def __post_init__(self) -> None:
        targets = tuple(binding.target for binding in self.bindings)
        if (
            not self.bindings
            or tuple(sorted(targets)) != targets
            or len(targets) != len(set(targets))
        ):
            raise ValueError("parameter overlay bindings must be sorted and unique")


__all__ = [
    "ParameterModuleKind",
    "ParameterOverlayFieldBinding",
    "ParameterOverlayOperator",
    "ParameterOverlayPublication",
    "ParameterRole",
    "ParameterTarget",
]
