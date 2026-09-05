"""Closed logical bindings for immutable parameter-overlay publications."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.semantic_graph import AnchorId
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
class ParameterOverlayPinnedTensor:
    target: ParameterTarget
    ref: StateRef
    tensor: torch.Tensor


@dataclass(frozen=True, slots=True)
class ParameterOverlayRowAssignment:
    identity: RequestIdentity
    target: ParameterTarget
    ref: StateRef
    phase: HiddenPhase
    packed_rows: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.phase not in (HiddenPhase.PREFILL, HiddenPhase.DECODE)
            or not self.packed_rows
            or tuple(sorted(set(self.packed_rows))) != self.packed_rows
        ):
            raise ValueError("parameter overlay row assignment is not canonical")


@dataclass(frozen=True, slots=True)
class ParameterOverlayStepPlan:
    plan_epoch: int
    apply_anchor_id: AnchorId
    capture_anchor_id: AnchorId
    total_rows: int
    assignments: tuple[ParameterOverlayRowAssignment, ...]

    def __post_init__(self) -> None:
        assigned = tuple(
            (assignment.target, row)
            for assignment in self.assignments
            for row in assignment.packed_rows
        )
        canonical_assignments = tuple(
            sorted(
                self.assignments,
                key=lambda item: (
                    item.target,
                    item.identity.request_id,
                    item.identity.prompt_idx,
                    item.identity.sample_idx,
                    item.phase.value,
                    item.packed_rows,
                ),
            )
        )
        if (
            self.plan_epoch < 1
            or int(self.apply_anchor_id) < 0
            or int(self.capture_anchor_id) < int(self.apply_anchor_id)
            or self.total_rows < 1
            or canonical_assignments != self.assignments
            or len(assigned) != len(set(assigned))
            or any(row < 0 or row >= self.total_rows for _target, row in assigned)
        ):
            raise ValueError("parameter overlay step plan is not canonical")


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
    "ParameterOverlayPinnedTensor",
    "ParameterOverlayRowAssignment",
    "ParameterOverlayStepPlan",
    "ParameterRole",
    "ParameterTarget",
    "CapturedParameterOperand",
]
