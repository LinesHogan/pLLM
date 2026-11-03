"""Closed backend attachment inputs shared by planning and worker installation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_hidden import PackedHiddenResourceBinding
from pllm.contracts.parameter_overlay import FieldView, ParameterOverlayFieldBinding, ProjectionShape
from pllm.contracts.sampler import SamplerStage
from pllm.contracts.semantic_graph import StateResource, TaskSymbol
from pllm.contracts.state_representation import (
    StateDeviceKind,
)
from pllm.contracts.tensor_layout import TensorDType


class AttachmentGraphMode(StrEnum):
    EAGER = "eager"
    CUDA_GRAPH = "cuda_graph"


@dataclass(frozen=True, slots=True)
class AttachmentHiddenResource:
    binding: PackedHiddenResourceBinding
    phase: HiddenPhase

    @property
    def resource_key(self) -> tuple[str, str, str]:
        return self.binding.locator_key


@dataclass(frozen=True, slots=True)
class CausalModelDomain:
    requests: int
    layers: int
    positions: int


@dataclass(frozen=True, slots=True)
class ModelProbe:
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
    # The distinct absolute layers whose logits the layer-logits input holds, in its column order.
    layer_logits: tuple[int, ...] = ()
    # Opaque library scratch measured at model load; CPU reports exclude it explicitly.
    blas_workspace_bytes: int = 0
    # Shape-only planning selects an ordinal without querying CUDA; workers use their current device.
    device_index: int | None = None


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

    resource: StateResource
    layer: int
    fields: tuple[FieldView, ...]
    first_tokens: int
    last_tokens: int
    operator: ActivationPatchOperator = ActivationPatchOperator.LOW_RANK_AFFINE
    decode: DecodeWindow = DecodeWindow.NONE
    decode_refresh: DecodeRefresh = DecodeRefresh.INHERIT
    on_missing: MissingPolicy = MissingPolicy.STOP


@dataclass(frozen=True, slots=True)
class SamplerBinding:
    """The method whose result enters vLLM's sampler, and at which stage."""

    task: TaskSymbol
    stage: SamplerStage


@dataclass(frozen=True, slots=True)
class BackendAttachmentSpec:
    """What a workload does to the model's step, one independent entry per kind of effect.

    Effects compose by appearing side by side; each binding names its own tasks and states. The planner orders
    their tasks; the backend installs each model-side binding on its own hook.
    """

    sampler: tuple[SamplerBinding, ...] = ()
    # The @pllm.train methods (request rows or a host batch).
    training: tuple[TaskSymbol, ...] = ()
    exports: tuple[TaskSymbol, ...] = ()
    # Every pllm.model.Adapter and @pllm.train_in_forward projection, each naming the state it reads.
    overlay_bindings: tuple[ParameterOverlayFieldBinding, ...] = ()
    # The @pllm.train_in_forward transition the projection hooks drive.
    in_forward: TaskSymbol | None = None
    activation_bindings: tuple[ActivationPatchBinding, ...] = ()
    # The task that runs the @pllm.custom_forward programs (the in-forward transition when both are declared).
    custom_forward: TaskSymbol | None = None

    @property
    def empty(self) -> bool:
        return self == BackendAttachmentSpec()


@dataclass(frozen=True, slots=True)
class BackendAttachment:
    """Declared effects and the plan they were scheduled in, shared by the driver and worker."""

    spec: BackendAttachmentSpec
    probe: ModelProbe
    plan: PlanBundle


__all__ = [
    "BackendAttachmentSpec",
    "BackendAttachment",
    "AttachmentGraphMode",
    "AttachmentHiddenResource",
    "SamplerBinding",
    "SamplerStage",
    "ModelProbe",
]


TOKEN_CAPTURE_CAPABILITY = "execution/token_capture@1"
