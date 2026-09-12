"""Closed backend attachment facts bound to canonical physical plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.parameter_overlay import ParameterOverlayFieldBinding
from pllm.contracts.semantic_graph import (
    AnchorId,
    ImplementationId,
    TaskId,
    AnchorSymbol,
    StateResource,
    TaskSymbol,
)
from pllm.contracts.state_representation import (
    StateDeviceKind,
)
from pllm.contracts.tensor_layout import TensorDType
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.contracts.region_imports import RegionTensorImport
from pllm.contracts.region_composition import RegionFunctionBody


class AttachmentTopology(StrEnum):
    IN_PROCESS = "in_process"
    ENGINE_CORE = "engine_core"


class AttachmentKind(StrEnum):
    OBSERVATION = "observation"
    MODEL_REPLAY = "model_replay"
    SAMPLER_TRAINING = "sampler_training"
    PARAMETER_OVERLAY = "parameter_overlay"
    ACTIVATION_PATCH = "activation_patch"


class AttachmentGraphMode(StrEnum):
    EAGER = "eager"


class AttachmentCleanupStage(StrEnum):
    REQUEST_FINISH = "request_finish"


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
class CausalModelDomain:
    requests: int
    layers: int
    positions: int

    def __post_init__(self) -> None:
        if any(type(size) is not int or not 0 < size < 2**63 for size in (self.requests, self.layers, self.positions)):
            raise ValueError("causal model domain requires positive bounded index extents")


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
    schema_version: int = 2
    causal_domain: CausalModelDomain | None = None
    region_imports: tuple[RegionTensorImport, ...] = ()
    region_bodies: tuple[RegionFunctionBody, ...] = ()


class ActivationPatchOperator(StrEnum):
    LOW_RANK_AFFINE = "low_rank_affine"
    ADD = "add"


@dataclass(frozen=True, slots=True)
class ActivationPatchBinding:
    """Signed tensor operation on selected prefill block-output rows."""

    layer: int
    fields: tuple[LayoutFieldId, ...]
    first_tokens: int
    last_tokens: int
    operator: ActivationPatchOperator = ActivationPatchOperator.LOW_RANK_AFFINE


@dataclass(frozen=True, slots=True)
class BackendAttachmentSpec:
    kind: AttachmentKind
    tasks: tuple[tuple[AttachmentTaskRole, TaskSymbol], ...]
    sampler_anchor: AnchorSymbol
    state_resource: StateResource
    overlay_bindings: tuple[ParameterOverlayFieldBinding, ...] = ()
    activation_bindings: tuple[ActivationPatchBinding, ...] = ()
    sampler_stage: SamplerStage = SamplerStage.PACKED_POST_FILTER_EXACT


@dataclass(frozen=True, slots=True)
class BackendAttachmentRequest:
    attachment_kind: AttachmentKind
    packed_row_schema_version: int
    task_bindings: tuple[AttachmentTaskBinding, ...]
    sampler_anchor_id: AnchorId
    sampler_anchor_symbol: str
    stream_dependencies: tuple[StreamDependency, ...]
    sampler_stage: SamplerStage
    cleanup_stage: AttachmentCleanupStage
    state_resource: str
    overlay_bindings: tuple[ParameterOverlayFieldBinding, ...] = ()
    activation_bindings: tuple[ActivationPatchBinding, ...] = ()
    schema_version: int = 7


@dataclass(frozen=True, slots=True)
class BackendAttachmentCompilerInputs:
    request: BackendAttachmentRequest
    probe: BackendPlacementProbe
    plan: PlanBundle


@dataclass(frozen=True, slots=True)
class BackendAttachmentDecision:
    request_digest: str
    signature: str
    schema_version: int = 7


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
    schema_version: int = 7


@dataclass(frozen=True, slots=True)
class SignedBackendAttachment:
    inputs: BackendAttachmentCompilerInputs
    decision: BackendAttachmentDecision
    physical: BackendAttachmentPhysicalPlan


__all__ = [
    "BackendAttachmentSpec",
    "SignedBackendAttachment",
    "AttachmentCleanupStage",
    "AttachmentGraphMode",
    "AttachmentKind",
    "AttachmentHiddenResource",
    "SamplerStage",
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
