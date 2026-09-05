"""Closed backend attachment facts bound to canonical physical plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.semantic_graph import (
    AnchorId,
    ImplementationId,
    TaskId,
    AnchorSymbol,
    StateResource,
    TaskSymbol,
)
from tllm.contracts.state_representation import (
    StateDeviceKind,
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

    @property
    def resource_key(self) -> tuple[str, str, str]:
        return self.binding.locator_key


@dataclass(frozen=True, slots=True)
class AttachmentTaskBinding:
    role: AttachmentTaskRole
    task_id: TaskId
    implementation_id: ImplementationId
    anchor_id: AnchorId


@dataclass(frozen=True, slots=True)
class AttachmentOverlayStageAnchor:
    stage: AttachmentOverlayStage
    anchor_id: AnchorId
    anchor_symbol: str


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
class RequestSamplerAttachmentSpec:
    compute_task: TaskSymbol
    transition_task: TaskSymbol
    sampler_anchor: AnchorSymbol
    state_resource: StateResource


@dataclass(frozen=True, slots=True)
class BackendAttachmentRequest:
    attachment_kind: AttachmentKind
    packed_row_schema_version: int
    task_bindings: tuple[AttachmentTaskBinding, ...]
    sampler_anchor_id: AnchorId
    sampler_anchor_symbol: str
    stream_dependencies: tuple[StreamDependency, ...]
    sampler_stage: AttachmentSamplerStage
    cleanup_stage: AttachmentCleanupStage
    state_resource: str
    overlay_field_ids: tuple[int, ...] = ()
    overlay_stage_anchors: tuple[AttachmentOverlayStageAnchor, ...] = ()
    schema_version: int = 2


@dataclass(frozen=True, slots=True)
class BackendAttachmentCompilerInputs:
    request: BackendAttachmentRequest
    probe: BackendPlacementProbe
    plan: CanonicalPlanBundle


@dataclass(frozen=True, slots=True)
class BackendAttachmentDecision:
    request_digest: str
    signature: str
    schema_version: int = 2


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
    schema_version: int = 2


@dataclass(frozen=True, slots=True)
class SignedBackendAttachment:
    inputs: BackendAttachmentCompilerInputs
    decision: BackendAttachmentDecision
    physical: BackendAttachmentPhysicalPlan


__all__ = [
    "RequestSamplerAttachmentSpec",
    "SignedBackendAttachment",
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
