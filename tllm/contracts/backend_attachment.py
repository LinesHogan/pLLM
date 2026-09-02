"""Closed backend attachment facts bound to canonical physical plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.contracts.request_identity import (
    RequestKeyField,
)
from tllm.contracts.semantic_graph import (
    AnchorId,
    ImplementationId,
    TaskId,
    PhysicalPlan,
    ScheduleDecision,
    SchedulerTimeline,
    SemanticGraph,
)
from tllm.contracts.state_representation import (
    StateDeviceKind,
    StatePhysicalPlan,
    StateRepresentationDecision,
    StateRepresentationRequest,
)
from tllm.contracts.tensor_layout import TensorDType


class AttachmentTopology(StrEnum):
    IN_PROCESS = "in_process"


class AttachmentKind(StrEnum):
    REQUEST_SAMPLER_TRAINING = "request_sampler_training"
    DENSE_PARAMETER_OVERLAY = "dense_parameter_overlay"


class AttachmentGraphMode(StrEnum):
    EAGER = "eager"


class AttachmentSamplerStage(StrEnum):
    PACKED_POST_FILTER_EXACT = "packed_post_filter_exact"


class AttachmentCleanupStage(StrEnum):
    REQUEST_FINISH = "request_finish"


class AttachmentOverlayStage(StrEnum):
    PIN = "pin"
    APPLY = "apply"
    CAPTURE = "capture"
    TRANSITION = "transition"
    PUBLISH = "publish"
    RELEASE = "release"
    FINISH = "finish"


class StreamDependency(StrEnum):
    HIDDEN_PRODUCER_TO_COMPUTE = "hidden_producer_to_compute"
    COMPUTE_TO_SAMPLER = "compute_to_sampler"
    SAMPLER_TO_TRANSITION = "sampler_to_transition"
    TRANSITION_TO_RELEASE = "transition_to_release"


class AttachmentTaskRole(StrEnum):
    COMPUTE = "compute"
    TRANSITION = "transition"
    PUBLISH = "publish"


@dataclass(frozen=True, slots=True)
class AttachmentHiddenResource:
    binding: PackedHiddenResourceBinding
    phase: HiddenPhase

    def __post_init__(self) -> None:
        if (
            type(self.binding) is not PackedHiddenResourceBinding
            or type(self.phase) is not HiddenPhase
            or self.phase is HiddenPhase.NONE
        ):
            raise ValueError("attachment hidden resource is not closed")

    @property
    def resource_key(self) -> tuple[str, str, str]:
        return self.binding.locator_key


@dataclass(frozen=True, slots=True)
class AttachmentTaskBinding:
    role: AttachmentTaskRole
    task_id: TaskId
    implementation_id: ImplementationId
    anchor_id: AnchorId

    def __post_init__(self) -> None:
        if (
            type(self.role) is not AttachmentTaskRole
            or type(self.task_id) is not int
            or type(self.implementation_id) is not str
            or not self.implementation_id
            or type(self.anchor_id) is not int
            or int(self.task_id) < 0
            or int(self.anchor_id) < 0
        ):
            raise ValueError("attachment task binding is invalid")


@dataclass(frozen=True, slots=True)
class AttachmentOverlayStageAnchor:
    stage: AttachmentOverlayStage
    anchor_id: AnchorId
    anchor_symbol: str

    def __post_init__(self) -> None:
        if (
            type(self.stage) is not AttachmentOverlayStage
            or type(self.anchor_id) is not int
            or int(self.anchor_id) < 0
            or type(self.anchor_symbol) is not str
            or not self.anchor_symbol
        ):
            raise ValueError("attachment overlay stage anchor is invalid")


@dataclass(frozen=True, slots=True)
class BackendPlacementProbe:
    adapter_id: str
    backend_abi_digest: str
    runner_fingerprint: str
    model_fingerprint: str
    topology: AttachmentTopology
    graph_mode: AttachmentGraphMode
    device_kind: StateDeviceKind
    device_ordinal: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    tensor_dtype: TensorDType
    hidden_size: int
    max_packed_rows: int
    hidden_resources: tuple[AttachmentHiddenResource, ...]
    capabilities: tuple[str, ...]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class BackendAttachmentRequest:
    adapter_id: str
    attachment_kind: AttachmentKind
    backend_abi_digest: str
    runner_fingerprint: str
    model_fingerprint: str
    topology: AttachmentTopology
    graph_mode: AttachmentGraphMode
    device_kind: StateDeviceKind
    device_ordinal: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    tensor_dtype: TensorDType
    hidden_size: int
    max_packed_rows: int
    packed_row_schema_version: int
    hidden_resources: tuple[AttachmentHiddenResource, ...]
    task_bindings: tuple[AttachmentTaskBinding, ...]
    sampler_anchor_id: AnchorId
    sampler_anchor_symbol: str
    stream_dependencies: tuple[StreamDependency, ...]
    sampler_stage: AttachmentSamplerStage
    cleanup_stage: AttachmentCleanupStage
    request_key_fields: tuple[RequestKeyField, ...]
    state_resource: str
    plan_epoch: int
    schedule_signature: str
    state_request_digest: str
    state_representation_signature: str
    execution_parameters_digest: str
    capabilities: tuple[str, ...]
    overlay_field_ids: tuple[int, ...] = ()
    overlay_stage_anchors: tuple[AttachmentOverlayStageAnchor, ...] = ()
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class BackendAttachmentCompilerInputs:
    request: BackendAttachmentRequest
    probe: BackendPlacementProbe
    graph: SemanticGraph
    timeline: SchedulerTimeline
    schedule_decision: ScheduleDecision
    schedule_physical: PhysicalPlan
    state_request: StateRepresentationRequest
    state_decision: StateRepresentationDecision
    state_physical: StatePhysicalPlan

    def __post_init__(self) -> None:
        if (
            type(self.request) is not BackendAttachmentRequest
            or type(self.probe) is not BackendPlacementProbe
            or type(self.graph) is not SemanticGraph
            or type(self.timeline) is not SchedulerTimeline
            or type(self.schedule_decision) is not ScheduleDecision
            or type(self.schedule_physical) is not PhysicalPlan
            or type(self.state_request) is not StateRepresentationRequest
            or type(self.state_decision) is not StateRepresentationDecision
            or type(self.state_physical) is not StatePhysicalPlan
        ):
            raise TypeError("backend attachment compiler inputs must be exact plans")


@dataclass(frozen=True, slots=True)
class BackendAttachmentDecision:
    request_digest: str
    signature: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if any(
            type(value) is not str
            for value in (self.request_digest, self.signature)
        ):
            raise TypeError("attachment decision digests must be strings")
        for value in (self.request_digest, self.signature):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("attachment decision digests must be SHA-256")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("attachment decision schema is unsupported")


@dataclass(frozen=True, slots=True)
class BackendAttachmentPhysicalPlan:
    plan_epoch: int
    adapter_id: str
    attachment_signature: str
    schedule_signature: str
    state_representation_signature: str
    device_kind: StateDeviceKind
    device_ordinal: int
    topology: AttachmentTopology
    graph_mode: AttachmentGraphMode
    bundle_signature: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        signature_values = (
            self.attachment_signature,
            self.schedule_signature,
            self.state_representation_signature,
            self.bundle_signature,
        )
        if any(type(value) is not str for value in signature_values):
            raise TypeError("physical attachment signatures must be strings")
        for value in signature_values:
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("physical attachment signatures must be SHA-256")
        if (
            type(self.plan_epoch) is not int
            or self.plan_epoch < 1
            or type(self.adapter_id) is not str
            or not self.adapter_id
            or type(self.device_kind) is not StateDeviceKind
            or type(self.device_ordinal) is not int
            or self.device_ordinal < 0
            or type(self.topology) is not AttachmentTopology
            or type(self.graph_mode) is not AttachmentGraphMode
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("physical attachment plan is invalid")


__all__ = [
    "AttachmentCleanupStage",
    "AttachmentGraphMode",
    "AttachmentKind",
    "AttachmentHiddenResource",
    "AttachmentOverlayStage",
    "AttachmentOverlayStageAnchor",
    "AttachmentSamplerStage",
    "AttachmentTaskBinding",
    "AttachmentTaskRole",
    "AttachmentTopology",
    "BackendAttachmentDecision",
    "BackendAttachmentCompilerInputs",
    "BackendAttachmentPhysicalPlan",
    "BackendAttachmentRequest",
    "BackendPlacementProbe",
    "StreamDependency",
]
