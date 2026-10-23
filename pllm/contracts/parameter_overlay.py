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
class Exact:
    """Each prefill chunk reads the preceding chunk's committed adapter."""


class KVPolicy(StrEnum):
    """Which parameter history the model's cached keys and values represent."""

    CAUSAL_HISTORY = "causal_history"


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


@dataclass(frozen=True)
class ModelScan:
    """Learn once per nonempty prefill chunk, including its tail; freeze for decode."""

    chunk_size: int
    apply: ParameterEffect
    learn: str
    algorithm: Exact = Exact()
    kv: KVPolicy = KVPolicy.CAUSAL_HISTORY

    def __post_init__(self) -> None:
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("ModelScan needs a positive chunk size")
        if not self.learn:
            raise ValueError("ModelScan needs a training method")
        if type(self.algorithm) is not Exact or self.kv is not KVPolicy.CAUSAL_HISTORY:
            raise ValueError("ModelScan currently supports Exact with CAUSAL_HISTORY")


__all__ = [
    "ModelScan",
    "ParameterEffect",
    "Exact",
    "KVPolicy",
    "ParameterModuleKind",
    "ParameterOverlayFieldBinding",
    "ParameterOverlayOperator",
    "ParameterRole",
    "ParameterTarget",
    "CapturedParameterOperand",
]
