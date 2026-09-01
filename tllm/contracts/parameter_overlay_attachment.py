"""Closed canonical contracts for signed parameter-overlay attachment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tllm.contracts.parameter_overlay import (
    ParameterOverlayOperator,
    ParameterTarget,
)
from tllm.contracts.request_identity import (
    RequestKeyField,
    canonical_request_key_fields,
)
from tllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    ImplementationId,
    PhysicalPlan,
    ScheduleDecision,
    SchedulerTimeline,
    SemanticGraph,
    TaskId,
)
from tllm.contracts.state_representation import (
    StatePhysicalPlan,
    StateRepresentationDecision,
    StateRepresentationRequest,
)
from tllm.contracts.tensor_layout import LayoutFieldId
from tllm.contracts.tensor_layout import TensorLayoutDecision, TensorLayoutRequest


class OperandCaptureKind(StrEnum):
    MLP_DOWN_PROJECTION_PAIR = "mlp_down_projection_pair"


class ParameterOverlayStage(StrEnum):
    PIN = "pin"
    APPLY = "apply"
    CAPTURE = "capture"
    TRANSITION = "transition"
    PUBLISH = "publish"
    RELEASE = "release"
    FINISH = "finish"


class ParameterOverlayTaskRole(StrEnum):
    TRANSITION = "transition"
    PUBLISH = "publish"


def _digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True, slots=True, order=True)
class ParameterOverlayAttachmentBinding:
    target: ParameterTarget
    operator: ParameterOverlayOperator
    field_id: LayoutFieldId

    def __post_init__(self) -> None:
        if (
            self.operator is not ParameterOverlayOperator.DENSE_ADD
            or int(self.field_id) < 0
        ):
            raise ValueError("attachment binding must be a dense-add layout field")


@dataclass(frozen=True, slots=True, order=True)
class ParameterOverlayStageAnchor:
    stage: ParameterOverlayStage
    anchor_id: AnchorId
    anchor_symbol: AnchorSymbol

    def __post_init__(self) -> None:
        if int(self.anchor_id) < 0 or not self.anchor_symbol:
            raise ValueError("parameter-overlay stage anchor is invalid")


@dataclass(frozen=True, slots=True)
class ParameterOverlayTaskBinding:
    role: ParameterOverlayTaskRole
    task_id: TaskId
    implementation_id: ImplementationId
    anchor_id: AnchorId

    def __post_init__(self) -> None:
        if (
            int(self.task_id) < 0
            or not self.implementation_id
            or int(self.anchor_id) < 0
        ):
            raise ValueError("parameter-overlay task binding is invalid")


def _validate_bindings(
    bindings: tuple[ParameterOverlayAttachmentBinding, ...],
) -> None:
    if (
        not bindings
        or tuple(sorted(bindings)) != bindings
        or len(set(bindings)) != len(bindings)
    ):
        raise ValueError("parameter-overlay bindings must be sorted and unique")


def _validate_stages(stages: tuple[ParameterOverlayStageAnchor, ...]) -> None:
    if tuple(stage.stage for stage in stages) != tuple(ParameterOverlayStage):
        raise ValueError("parameter-overlay stages must be complete and ordered")


def _validate_tasks(bindings: tuple[ParameterOverlayTaskBinding, ...]) -> None:
    if tuple(binding.role for binding in bindings) != tuple(
        ParameterOverlayTaskRole
    ):
        raise ValueError("parameter-overlay task bindings must be complete")


def _validate_capabilities(capabilities: tuple[str, ...]) -> None:
    if (
        not capabilities
        or tuple(sorted(capabilities)) != capabilities
        or len(set(capabilities)) != len(capabilities)
        or any(not value for value in capabilities)
    ):
        raise ValueError("parameter-overlay capabilities are not canonical")


@dataclass(frozen=True, slots=True)
class ParameterOverlayAttachmentCandidate:
    candidate_id: str
    adapter_id: str
    backend_abi_digest: str
    runner_fingerprint: str
    model_fingerprint: str
    bindings: tuple[ParameterOverlayAttachmentBinding, ...]
    operand_capture: OperandCaptureKind
    stage_anchors: tuple[ParameterOverlayStageAnchor, ...]
    task_bindings: tuple[ParameterOverlayTaskBinding, ...]
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.candidate_id
            or not self.adapter_id
            or not all(
                _digest(value)
                for value in (
                    self.backend_abi_digest,
                    self.runner_fingerprint,
                    self.model_fingerprint,
                )
            )
        ):
            raise ValueError("parameter-overlay candidate is invalid")
        _validate_bindings(self.bindings)
        _validate_stages(self.stage_anchors)
        _validate_tasks(self.task_bindings)
        _validate_capabilities(self.capabilities)


@dataclass(frozen=True, slots=True)
class ParameterOverlayAttachmentRequest:
    adapter_id: str
    backend_abi_digest: str
    runner_fingerprint: str
    model_fingerprint: str
    state_resource: str
    request_key_fields: tuple[RequestKeyField, ...]
    bindings: tuple[ParameterOverlayAttachmentBinding, ...]
    operand_capture: OperandCaptureKind
    stage_anchors: tuple[ParameterOverlayStageAnchor, ...]
    task_bindings: tuple[ParameterOverlayTaskBinding, ...]
    plan_epoch: int
    graph_signature: str
    schedule_signature: str
    state_representation_signature: str
    capabilities: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not self.adapter_id
            or not self.state_resource
            or self.plan_epoch < 1
            or self.schema_version != 1
            or not all(
                _digest(value)
                for value in (
                    self.backend_abi_digest,
                    self.runner_fingerprint,
                    self.model_fingerprint,
                    self.graph_signature,
                    self.schedule_signature,
                    self.state_representation_signature,
                )
            )
        ):
            raise ValueError("parameter-overlay request is invalid")
        canonical_request_key_fields(self.request_key_fields)
        _validate_bindings(self.bindings)
        _validate_stages(self.stage_anchors)
        _validate_tasks(self.task_bindings)
        _validate_capabilities(self.capabilities)


@dataclass(frozen=True, slots=True)
class ParameterOverlayAttachmentInputs:
    request: ParameterOverlayAttachmentRequest
    candidates: tuple[ParameterOverlayAttachmentCandidate, ...]
    graph: SemanticGraph
    timeline: SchedulerTimeline
    schedule_decision: ScheduleDecision
    schedule_physical: PhysicalPlan
    state_request: StateRepresentationRequest
    state_decision: StateRepresentationDecision
    state_physical: StatePhysicalPlan
    tensor_layout_request: TensorLayoutRequest
    tensor_layout_decision: TensorLayoutDecision

    def __post_init__(self) -> None:
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if (
            not candidate_ids
            or candidate_ids != tuple(sorted(candidate_ids))
            or len(set(candidate_ids)) != len(candidate_ids)
        ):
            raise ValueError("parameter-overlay candidates must be sorted and nonempty")


@dataclass(frozen=True, slots=True)
class ParameterOverlayAttachmentDecision:
    request_digest: str
    selected_candidate_id: str
    signature: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            not self.selected_candidate_id
            or not _digest(self.request_digest)
            or not _digest(self.signature)
            or self.schema_version != 1
        ):
            raise ValueError("parameter-overlay decision is invalid")


@dataclass(frozen=True, slots=True)
class ParameterOverlayAttachmentPhysicalPlan:
    plan_epoch: int
    selected_candidate_id: str
    attachment_signature: str
    schedule_signature: str
    state_representation_signature: str
    bindings: tuple[ParameterOverlayAttachmentBinding, ...]
    stage_anchors: tuple[ParameterOverlayStageAnchor, ...]
    task_bindings: tuple[ParameterOverlayTaskBinding, ...]
    bundle_signature: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if (
            self.plan_epoch < 1
            or not self.selected_candidate_id
            or not all(
                _digest(value)
                for value in (
                    self.attachment_signature,
                    self.schedule_signature,
                    self.state_representation_signature,
                    self.bundle_signature,
                )
            )
            or self.schema_version != 1
        ):
            raise ValueError("parameter-overlay physical plan is invalid")
        _validate_bindings(self.bindings)
        _validate_stages(self.stage_anchors)
        _validate_tasks(self.task_bindings)


__all__ = [
    "OperandCaptureKind",
    "ParameterOverlayAttachmentBinding",
    "ParameterOverlayAttachmentCandidate",
    "ParameterOverlayAttachmentDecision",
    "ParameterOverlayAttachmentInputs",
    "ParameterOverlayAttachmentPhysicalPlan",
    "ParameterOverlayAttachmentRequest",
    "ParameterOverlayStage",
    "ParameterOverlayStageAnchor",
    "ParameterOverlayTaskBinding",
    "ParameterOverlayTaskRole",
]
