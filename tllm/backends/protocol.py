#!/usr/bin/env python3
"""Backend driver contracts for binding the tLLM runtime to an inference engine.

The runtime core (ports, bundles, dispatch plan) is backend-agnostic. A
``BackendDriver`` owns every interaction with engine-private structure:
runner classes, hook installation, sampler interception and lifecycle
boundaries. Detection is structural (class/module/signature probes), not
version-string-only, and unknown structures fail fast instead of falling
through to a "future versions probably look the same" branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from tllm.contracts.backend_capability import BackendCapability
from tllm.contracts.hidden_locator import HiddenLocatorCapability
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.request_identity import RequestFinishedBatch
from tllm.contracts.typed_replay import TypedLayerReplayHandle


class BackendObservableEvent(StrEnum):
    """Closed backend hook events admitted by a signed timeline."""

    PREPARE_INPUTS_PRE = "prepare_inputs.pre"
    PREPARE_INPUTS_POST = "prepare_inputs.post"
    EXECUTE_MODEL_PRE = "execute_model.pre"
    EXECUTE_MODEL_POST = "execute_model.post"
    SAMPLE_TOKENS_PRE = "sample_tokens.pre"
    SAMPLER_PRE_DRAW = "sampler.pre_draw"
    SAMPLE_TOKENS_POST = "sample_tokens.post"


def runner_model_module(runner: object) -> "torch.nn.Module":
    """Resolve an unwrapped model at the external-runner protocol boundary."""

    import torch

    get_model = getattr(runner, "get_model", None)
    model = get_model() if callable(get_model) else None
    if not isinstance(model, torch.nn.Module):
        model = getattr(runner, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("inference runner must expose an unwrapped torch.nn.Module")
    return model


def runner_device(runner: object) -> "torch.device":
    """Resolve the runner device at the external protocol boundary."""

    import torch

    device = getattr(runner, "device", None)
    if not isinstance(device, torch.device):
        raise TypeError("inference runner must expose a torch.device")
    return device

if TYPE_CHECKING:
    import torch
    from tllm.contracts.packed_hidden import (
        PackedHiddenResourceBinding,
        PackedHiddenSnapshotBatch,
    )
    from tllm.contracts.packed_rows import PackedPhaseRowBatch
    from tllm.runtime.parameter_overlay import (
        OverlayChannel,
        DenseOperandCapture,
        OverlayStepPlan,
        OverlayTarget,
        OverlayVersion,
        OverlayVersionRef,
        OverlayVersionStore,
    )
    from tllm.schedule import MainWorkloadTimeline


class PackedHiddenSnapshotSink(Protocol):
    def stage_packed_hidden_snapshots(
        self, batch: "PackedHiddenSnapshotBatch"
    ) -> None: ...


class PackedHiddenCaptureBinding(Protocol):
    @property
    def plan_epoch(self) -> int: ...

    @property
    def resources(self) -> tuple["PackedHiddenResourceBinding", ...]: ...

    def begin(self, row_batch: "PackedPhaseRowBatch") -> None: ...

    def publish(
        self, execution: PackedHiddenSnapshotSink | None = None
    ) -> "PackedHiddenSnapshotBatch": ...

    def finish(self) -> None: ...

    def abort_if_active(self) -> None: ...

    def close(self) -> None: ...


class SamplerStage(Enum):
    """Semantic stages inside one sampling pass that a backend can expose."""

    RAW_LOGITS = auto()
    AFTER_BIAS_AND_PENALTIES = auto()
    AFTER_TEMPERATURE = auto()
    AFTER_MIN_P = auto()
    AFTER_TOP_K_TOP_P = auto()
    BEFORE_DRAW = auto()
    AFTER_DRAW = auto()


class GraphSafety(Enum):
    """How a hook/producer behaves with respect to CUDA graph capture/replay."""

    EAGER_ONLY = auto()
    CAPTURE_SAFE = auto()
    REPLAY_STATEFUL = auto()


class DenseOverlayTargetHandle(Protocol):
    @property
    def target(self) -> OverlayTarget: ...

    @property
    def apply_count(self) -> int: ...

    @property
    def version_count(self) -> int: ...

    @property
    def capture_expected(self) -> bool: ...

    def has_version(self, ref: OverlayVersionRef) -> bool: ...

    def install_version(
        self,
        version: OverlayVersion,
        tensors: Mapping[str, torch.Tensor],
    ) -> None: ...

    def retain_versions(self, refs: frozenset[OverlayVersionRef]) -> None: ...

    def prepare_step(self, plan: OverlayStepPlan) -> None: ...

    def clear_step(self) -> None: ...

    def take_capture(self) -> DenseOperandCapture | None: ...

    def restore(self) -> None: ...


@dataclass(slots=True, init=False)
class OverlayBackendState:
    """Runner-owned installation state shared across overlay store epochs."""

    stores: Mapping[OverlayChannel, OverlayVersionStore]
    version_handles: dict[OverlayVersionRef, int]
    request_identities: dict[str, RequestIdentity]
    attached_requests: set[str]
    dense_request_ids: set[str]
    dense_targets: dict[OverlayTarget, DenseOverlayTargetHandle]
    dense_reported_rows: dict[OverlayTarget, int]
    dense_targets_validated: bool
    next_handle_ordinal: int

    def __init__(self) -> None:
        self.stores = {}
        self.version_handles = {}
        self.request_identities = {}
        self.attached_requests = set()
        self.dense_request_ids = set()
        self.dense_targets = {}
        self.dense_reported_rows = {}
        self.dense_targets_validated = False
        self.next_handle_ordinal = 0

    def owns_stores(
        self, stores: Mapping[OverlayChannel, OverlayVersionStore]
    ) -> bool:
        return self.stores.keys() == stores.keys() and all(
            self.stores[channel] is store for channel, store in stores.items()
        )

    def rebind(
        self, stores: Mapping[OverlayChannel, OverlayVersionStore]
    ) -> None:
        self.stores = MappingProxyType(dict(stores))
        self.version_handles.clear()
        self.request_identities.clear()
        self.attached_requests.clear()
        self.dense_request_ids.clear()
        self.dense_targets.clear()
        self.dense_reported_rows.clear()
        self.dense_targets_validated = False

    def allocate_handle_ordinal(self) -> int:
        self.next_handle_ordinal += 1
        return self.next_handle_ordinal


@dataclass(frozen=True)
class VersionedParameterOverlayCapabilities:
    """Generic immutable overlay installation and request-lease support."""

    representations: frozenset[str]
    scopes: frozenset[str]
    phases: frozenset[str]
    pinning: frozenset[str]
    mixed_versions: bool
    operand_captures: frozenset[str] = frozenset()
    graph_safety: GraphSafety = GraphSafety.EAGER_ONLY
    tensor_parallel_sizes: frozenset[int] = frozenset({1})

    def capability_names(self) -> frozenset[str]:
        return frozenset(
            {
                *(f"parameter_overlay/representation/{value}" for value in self.representations),
                *(f"parameter_overlay/scope/{value}" for value in self.scopes),
                *(f"parameter_overlay/phase/{value}" for value in self.phases),
                *(f"parameter_overlay/pinning/{value}" for value in self.pinning),
                *(
                    f"parameter_overlay/operand_capture/{value}"
                    for value in self.operand_captures
                ),
                *(
                    f"parameter_overlay/tensor_parallel_size/{value}"
                    for value in self.tensor_parallel_sizes
                ),
                f"parameter_overlay/mixed_versions/{str(self.mixed_versions).lower()}",
            }
        )


@dataclass(frozen=True)
class OrderedRowFusionCapabilities:
    """Dependence effects a backend can satisfy inside one ordered-row call."""

    internalized_edges: frozenset[str]
    max_rows: int = 2

    def __post_init__(self) -> None:
        if not self.internalized_edges or any(not value for value in self.internalized_edges):
            raise TypeError("internalized_edges must be a non-empty frozenset[str]")
        if self.max_rows < 2:
            raise ValueError("ordered row fusion max_rows must be >= 2")


ORDERED_ROW_FUSION_POLICY_PREFIX = "model_region/ordered_row_fusion_policy/v1/"


def ordered_row_fusion_policy_fingerprint(
    fusion: OrderedRowFusionCapabilities | None,
) -> str:
    """Return the canonical policy signed with a native Region epoch."""
    if fusion is None:
        return "none"
    return (
        f"max_rows={fusion.max_rows};"
        f"internalized_edges={','.join(sorted(fusion.internalized_edges))}"
    )


@dataclass(frozen=True)
class ModelRegionCapabilities:
    phases: frozenset[str]
    replay: bool
    kv_snapshot: bool
    kv_overwrite: bool
    ordered_row_fusion: OrderedRowFusionCapabilities | None = None
    graph_safety: GraphSafety = GraphSafety.EAGER_ONLY
    tensor_parallel_sizes: frozenset[int] = frozenset({1})

    def capability_names(self) -> frozenset[str]:
        names = {
            f"model_region/replay/{phase}" for phase in self.phases if self.replay
        }
        if self.kv_snapshot:
            names.add("model_region/kv_snapshot")
        if self.kv_overwrite:
            names.add("model_region/kv_overwrite")
        if self.ordered_row_fusion is not None:
            names.add("model_region/ordered_row_fusion")
        names.add(
            ORDERED_ROW_FUSION_POLICY_PREFIX
            + ordered_row_fusion_policy_fingerprint(self.ordered_row_fusion)
        )
        return frozenset(names)


@dataclass(frozen=True)
class BackendCapabilities:
    backend_id: str
    runner_family: str  # "mrv1" | "mrv2"
    sampler_stages: frozenset[SamplerStage]
    native_min_p: bool
    tuple_sampler_return: bool
    split_execute_and_sample: bool
    graph_safety: GraphSafety = GraphSafety.EAGER_ONLY
    parameter_overlay: VersionedParameterOverlayCapabilities | None = None
    model_region: ModelRegionCapabilities | None = None
    extensions: frozenset[BackendCapability] = frozenset()
    eager_extensions: frozenset[BackendCapability] = frozenset()
    hidden_locators: frozenset[HiddenLocatorCapability] = frozenset()

    def capability_names(self) -> frozenset[str]:
        parameter_overlay = self.parameter_overlay
        region = self.model_region
        parameter_names: frozenset[str] = (
            frozenset()
            if parameter_overlay is None
            else parameter_overlay.capability_names()
        )
        region_names: frozenset[str] = (
            frozenset() if region is None else region.capability_names()
        )
        extension_names = frozenset(
            capability.value for capability in self.extensions
        )
        locator_names = frozenset(
            locator.value for locator in self.hidden_locators
        )
        return parameter_names | region_names | extension_names | locator_names

class RunnerBindingDriver(Protocol):
    """Driver surface retained by a live per-runner binding."""

    backend_id: str

    def capabilities(self) -> BackendCapabilities: ...

    def install_typed_layer_replay(
        self, *, runtime: Any, runner: Any
    ) -> TypedLayerReplayHandle: ...

    def restore_typed_layer_replay(self, *, runtime: Any) -> int: ...

    def consume_parameter_overlay_finished_batch(
        self,
        *,
        runtime: Any,
        runner: Any,
        state: OverlayBackendState,
        batch: RequestFinishedBatch,
    ) -> int: ...

    def synchronize_parameter_overlays(
        self,
        *,
        runtime: Any,
        runner: Any,
        state: OverlayBackendState,
    ) -> int: ...

    def runner_model_layer_count(self, runner: Any) -> int: ...

    def packed_row_plan_inputs(
        self,
        *,
        runner: Any,
        num_scheduled_tokens: Any | None,
        input_batch: Any | None,
    ) -> Any: ...


@dataclass
class RunnerBinding:
    """Per-runner-instance binding record."""

    driver: RunnerBindingDriver
    runner: Any
    capabilities: BackendCapabilities
    parameter_overlay_state: OverlayBackendState = field(
        default_factory=OverlayBackendState
    )
    typed_layer_replay_handle: TypedLayerReplayHandle | None = None
    packed_hidden_capture: PackedHiddenCaptureBinding | None = None
    active: bool = True


RUNNER_BINDING_ATTR = "_tllm_runner_binding"


def runner_binding_for(runner: Any) -> RunnerBinding | None:
    """Return the backend-neutral binding installed on a runner, if active."""
    binding = getattr(runner, RUNNER_BINDING_ATTR, None)
    return binding if isinstance(binding, RunnerBinding) else None


class BackendDriver(Protocol):
    """Owns engine-specific hook installation for one runner family."""

    backend_id: str

    def supports(self, info: Any) -> bool:
        """Structural check: can this driver drive the probed engine?"""
        ...

    def capabilities(self) -> BackendCapabilities:
        ...

    def export_main_workload_timeline(
        self,
        *,
        graph_mode: str,
        static_headroom_bytes: int,
    ) -> "MainWorkloadTimeline":
        """Return pinned backend regions and only genuinely observable anchors."""
        ...

    def install(self, *, core: Any, info: Any) -> None:
        """Install class-level hooks exactly once."""
        ...

    def unbind(self, binding: RunnerBinding) -> None:
        ...

    def consume_parameter_overlay_finished_batch(
        self,
        *,
        runtime: Any,
        runner: Any,
        state: OverlayBackendState,
        batch: RequestFinishedBatch,
    ) -> int: ...

    def synchronize_parameter_overlays(
        self,
        *,
        runtime: Any,
        runner: Any,
        state: OverlayBackendState,
    ) -> int: ...

    def install_typed_layer_replay(
        self,
        *,
        runtime: Any,
        runner: Any,
    ) -> TypedLayerReplayHandle:
        """Install (or return) the restorable typed replay hook for a runner.

        Consumer-owned binders reach the concrete backend replay machinery
        only through this protocol; they never import a driver module.
        """
        ...

    def restore_typed_layer_replay(self, *, runtime: Any) -> int:
        """Restore any typed replay hook installed for this runtime."""
        ...

    # The runtime must not inspect engine-private runner layouts.  Drivers
    # normalize the few fast-weight facts needed by the backend-neutral core.
    def runner_model_identity(self, runner: Any) -> int:
        ...

    def runner_active_request_count(self, runner: Any) -> int:
        ...

    def runner_model_layer_count(self, runner: Any) -> int:
        ...

    def runner_tensor_parallel_size(self, runner: Any) -> int:
        ...

    def packed_row_plan_inputs(
        self,
        *,
        runner: Any,
        num_scheduled_tokens: Any | None,
        input_batch: Any | None,
    ) -> Any:
        ...


class UnsupportedBackendError(RuntimeError):
    """Raised when no driver recognises the installed engine structure."""
