"""Closed backend attachment inputs shared by planning and worker installation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.parameter_overlay import FieldView, ParameterOverlayFieldBinding, ProjectionShape
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    ImplementationId,
    StateResource,
    TaskId,
    TaskSymbol,
)
from pllm.contracts.state_representation import (
    StateDeviceKind,
)
from pllm.contracts.tensor_layout import TensorDType


class AttachmentKind(StrEnum):
    OBSERVATION = "observation"
    CUSTOM_FORWARD = "custom_forward"
    REQUEST_TRAINING = "request_training"
    PARAMETER_OVERLAY = "parameter_overlay"
    ACTIVATION_PATCH = "activation_patch"


class AttachmentGraphMode(StrEnum):
    EAGER = "eager"
    CUDA_GRAPH = "cuda_graph"


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


@dataclass(frozen=True, slots=True)
class BackendPlacementProbe:
    model_fingerprint: str
    graph_mode: AttachmentGraphMode
    device_kind: StateDeviceKind
    tensor_dtype: TensorDType
    hidden_size: int
    max_packed_rows: int
    hidden_resources: tuple[AttachmentHiddenResource, ...]
    capabilities: tuple[str, ...]
    targets: tuple[ProjectionShape, ...] = ()
    causal_domain: CausalModelDomain | None = None
    position_states: tuple[tuple[str, float], ...] = ()


class ActivationPatchOperator(StrEnum):
    LOW_RANK_AFFINE = "low_rank_affine"
    ADD = "add"


class DecodeWindow(StrEnum):
    """Which decode rows a signed activation patch applies to."""

    NONE = "none"
    ALL = "all"


class DecodeRefresh(StrEnum):
    """INHERIT keeps the admission-pinned version; STEP selects the version at step entry."""

    INHERIT = "inherit"
    STEP = "step"


class MissingPolicy(StrEnum):
    """Declared behavior when a refreshing decode step has no new input version."""

    STOP = "stop"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class ActivationPatchBinding:
    """Signed tensor operation on selected prefill and declared decode block-output rows."""

    layer: int
    fields: tuple[FieldView, ...]
    first_tokens: int
    last_tokens: int
    operator: ActivationPatchOperator = ActivationPatchOperator.LOW_RANK_AFFINE
    decode: DecodeWindow = DecodeWindow.NONE
    decode_refresh: DecodeRefresh = DecodeRefresh.INHERIT
    on_missing: MissingPolicy = MissingPolicy.STOP


@dataclass(frozen=True, slots=True)
class BackendAttachmentSpec:
    kind: AttachmentKind
    tasks: tuple[tuple[AttachmentTaskRole, TaskSymbol], ...]
    sampler_anchor: AnchorSymbol
    state_resource: StateResource
    overlay_bindings: tuple[ParameterOverlayFieldBinding, ...] = ()
    activation_bindings: tuple[ActivationPatchBinding, ...] = ()
    sampler_stage: SamplerStage = SamplerStage.PACKED_POST_FILTER_EXACT

    @property
    def guides_sampler(self) -> bool:
        """True when a compute task feeds the sampler; False for training alone."""
        return self.kind is AttachmentKind.REQUEST_TRAINING and self.tasks[0][0] is AttachmentTaskRole.COMPUTE


@dataclass(frozen=True, slots=True)
class BackendAttachment:
    """Declared effects and resolved plan bindings shared by the driver and worker."""

    spec: BackendAttachmentSpec
    task_bindings: tuple[AttachmentTaskBinding, ...]
    sampler_anchor_id: AnchorId
    probe: BackendPlacementProbe
    plan: PlanBundle


__all__ = [
    "BackendAttachmentSpec",
    "BackendAttachment",
    "AttachmentGraphMode",
    "AttachmentKind",
    "AttachmentHiddenResource",
    "SamplerStage",
    "AttachmentTaskBinding",
    "AttachmentTaskRole",
    "BackendPlacementProbe",
]


TOKEN_CAPTURE_CAPABILITY = "execution/token_capture@1"
