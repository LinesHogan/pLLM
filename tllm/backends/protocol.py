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
from enum import Enum, auto
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from tllm.contracts.backend_capability import BackendCapability
from tllm.contracts.hidden_locator import HiddenLocatorCapability

if TYPE_CHECKING:
    from tllm.ports.module_weight_view import ModuleWeightViewLocator
    from tllm.schedule import MainWorkloadTimeline


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


@dataclass(frozen=True)
class ModuleWeightOverlayCapabilities:
    """Backend support matrix for ``ModuleWeightView`` writes."""

    modules: frozenset[str]
    write_modes: frozenset[str]
    phases: frozenset[str]
    request_keyed: bool
    graph_safety: GraphSafety = GraphSafety.EAGER_ONLY
    tensor_parallel_sizes: frozenset[int] = frozenset({1})

    def __post_init__(self) -> None:
        for field_name in ("modules", "write_modes", "phases"):
            values = getattr(self, field_name)
            if not isinstance(values, frozenset) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise TypeError(f"{field_name} must be frozenset[str]")
        if not isinstance(self.request_keyed, bool):
            raise TypeError("request_keyed must be bool")
        if not isinstance(self.graph_safety, GraphSafety):
            raise TypeError("graph_safety must be GraphSafety")
        if not isinstance(self.tensor_parallel_sizes, frozenset) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.tensor_parallel_sizes
        ):
            raise TypeError("tensor_parallel_sizes must be frozenset[int >= 1]")

    def capability_names(self) -> frozenset[str]:
        return frozenset(
            f"module_weight_view/{mode}/{phase}"
            for mode in self.write_modes
            for phase in self.phases
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
    module_weight_overlay: ModuleWeightOverlayCapabilities | None = None
    model_region: ModelRegionCapabilities | None = None
    extensions: frozenset[BackendCapability] = frozenset()
    hidden_locators: frozenset[HiddenLocatorCapability] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.extensions, frozenset) or any(
            not isinstance(capability, BackendCapability)
            for capability in self.extensions
        ):
            raise TypeError(
                "BackendCapabilities.extensions must be "
                "frozenset[BackendCapability]"
            )
        if not isinstance(self.hidden_locators, frozenset) or any(
            not isinstance(locator, HiddenLocatorCapability)
            for locator in self.hidden_locators
        ):
            raise TypeError(
                "BackendCapabilities.hidden_locators must be "
                "frozenset[HiddenLocatorCapability]"
            )

    def capability_names(self) -> frozenset[str]:
        overlay = self.module_weight_overlay
        region = self.model_region
        return (
            (frozenset() if overlay is None else overlay.capability_names())
            | (frozenset() if region is None else region.capability_names())
            | frozenset(capability.value for capability in self.extensions)
            | frozenset(locator.value for locator in self.hidden_locators)
        )


@dataclass(frozen=True)
class ResolvedModuleWeightView:
    """One backend-owned module target resolved from a logical locator."""

    module: Any
    layer: int
    module_name: str
    path: str


@dataclass
class RunnerBinding:
    """Per-runner-instance binding record."""

    driver: "BackendDriver"
    runner: Any
    capabilities: BackendCapabilities
    orig_methods: dict[str, Any] = field(default_factory=dict)
    module_weight_overlay_handles: dict[str, Any] = field(default_factory=dict)
    model_region_handles: dict[str, Any] = field(default_factory=dict)
    active: bool = True


_RUNNER_BINDING_ATTR = "_tllm_runner_binding"


def runner_binding_for(runner: Any) -> RunnerBinding | None:
    """Return the backend-neutral binding installed on a runner, if active."""
    binding = getattr(runner, _RUNNER_BINDING_ATTR, None)
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

    def resolve_module_weight_view(
        self,
        *,
        binding: RunnerBinding,
        locator: "ModuleWeightViewLocator",
    ) -> ResolvedModuleWeightView:
        ...

    def install_module_weight_overlay(
        self,
        *,
        binding: RunnerBinding,
        locator: "ModuleWeightViewLocator",
        installer: Callable[[object], Any],
    ) -> Any:
        ...

    def restore_module_weight_overlays(self, binding: RunnerBinding) -> int:
        ...

    def install_model_region_executor(
        self,
        *,
        binding: RunnerBinding,
        runtime: Any,
    ) -> Any:
        ...

    def restore_model_region_executors(self, binding: RunnerBinding) -> int:
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

    def fast_weight_packed_row_plan_inputs(
        self,
        *,
        runner: Any,
        num_scheduled_tokens: Any | None,
        input_batch: Any | None,
    ) -> Any:
        ...


class UnsupportedBackendError(RuntimeError):
    """Raised when no driver recognises the installed engine structure."""
