"""Backend-boundary lowering to the shared WorkGraph IR descriptors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from tllm.backends.protocol import (
    BackendCapabilities,
    BackendCapability,
    GraphSafety,
    SamplerStage,
)
from tllm.workgraph.graph_mode import GraphMode
from tllm.workgraph.ir import AnchorDesc, BackendSnapshot


class BackendCompatibilityError(RuntimeError):
    """Raised when a concrete backend is outside Phase 1a."""


def _require_str(value: object, label: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be str")
    if nonempty and not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _validate_capabilities(capabilities: object) -> BackendCapabilities:
    if not isinstance(capabilities, BackendCapabilities):
        raise TypeError("capabilities must be BackendCapabilities")
    _require_str(capabilities.backend_id, "BackendCapabilities.backend_id", nonempty=True)
    _require_str(capabilities.runner_family, "BackendCapabilities.runner_family", nonempty=True)
    sampler_stages = cast(object, capabilities.sampler_stages)
    if not isinstance(sampler_stages, frozenset):
        raise TypeError("BackendCapabilities.sampler_stages must be frozenset[SamplerStage]")
    typed_stages = cast(frozenset[object], sampler_stages)
    if any(not isinstance(stage, SamplerStage) for stage in typed_stages):
        raise TypeError("BackendCapabilities.sampler_stages must be frozenset[SamplerStage]")
    if not isinstance(cast(object, capabilities.native_min_p), bool):
        raise TypeError("BackendCapabilities.native_min_p must be bool")
    if not isinstance(cast(object, capabilities.tuple_sampler_return), bool):
        raise TypeError("BackendCapabilities.tuple_sampler_return must be bool")
    if not isinstance(cast(object, capabilities.split_execute_and_sample), bool):
        raise TypeError("BackendCapabilities.split_execute_and_sample must be bool")
    if not isinstance(cast(object, capabilities.graph_safety), GraphSafety):
        raise TypeError("BackendCapabilities.graph_safety must be GraphSafety")
    extensions = cast(object, capabilities.extensions)
    if not isinstance(extensions, frozenset):
        raise TypeError(
            "BackendCapabilities.extensions must be frozenset[BackendCapability]"
        )
    typed_extensions = cast(frozenset[object], extensions)
    if any(
        not isinstance(capability, BackendCapability)
        for capability in typed_extensions
    ):
        raise TypeError(
            "BackendCapabilities.extensions must be frozenset[BackendCapability]"
        )
    overlay = capabilities.module_weight_overlay
    if overlay is not None:
        from tllm.backends.protocol import ModuleWeightOverlayCapabilities

        if not isinstance(overlay, ModuleWeightOverlayCapabilities):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise TypeError(
                "BackendCapabilities.module_weight_overlay must be "
                "ModuleWeightOverlayCapabilities or None"
            )
    region = capabilities.model_region
    if region is not None:
        from tllm.backends.protocol import ModelRegionCapabilities

        if not isinstance(region, ModelRegionCapabilities):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise TypeError(
                "BackendCapabilities.model_region must be "
                "ModelRegionCapabilities or None"
            )
    return capabilities


@dataclass(frozen=True, slots=True)
class BackendResolution:
    """Concrete backend facts captured after vLLM resolves its graph mode.

    This is a pure-data boundary: it carries declared capabilities and an
    already-resolved graph-mode enum, never a runner, Tensor, stream, or
    backend-private handle.
    """

    capabilities: BackendCapabilities
    graph_mode: GraphMode
    backend_version: str = "0.27.1"
    tp_size: int = 1

    def __post_init__(self) -> None:
        capabilities = _validate_capabilities(self.capabilities)
        graph_mode = cast(object, self.graph_mode)
        if not isinstance(graph_mode, GraphMode):
            raise TypeError("BackendResolution.graph_mode must be GraphMode")
        if graph_mode not in _GRAPH_MODE_NAMES:
            raise ValueError(f"unsupported resolved graph mode {graph_mode!r}")
        backend_version = _require_str(
            self.backend_version,
            "BackendResolution.backend_version",
            nonempty=True,
        )
        if not backend_version.startswith("0.27."):
            raise ValueError("BackendResolution.backend_version must be in the vLLM 0.27.x family")
        if isinstance(self.tp_size, bool) or not isinstance(self.tp_size, int) or self.tp_size < 1:  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
            raise ValueError("BackendResolution.tp_size must be a positive int")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "graph_mode", graph_mode)
        object.__setattr__(self, "backend_version", backend_version)


_LOGICAL_HOOKS: tuple[str, ...] = (
    "prepare_inputs.pre",
    "prepare_inputs.post",
    "execute_model.pre",
    "execute_model.post",
    "sample_tokens.pre",
    "sample_tokens.post",
)

_GRAPH_MODE_NAMES: dict[GraphMode, str] = {
    GraphMode.EAGER: "eager",
    GraphMode.PIECEWISE: "piecewise",
    GraphMode.FULL: "full",
}


def _snapshot_from_capabilities(
    capabilities: BackendCapabilities,
    *,
    graph_mode: str,
    backend_version: str,
    allow_legacy_mode: bool,
) -> BackendSnapshot:
    capabilities = _validate_capabilities(capabilities)
    graph_mode = _require_str(graph_mode, "graph_mode", nonempty=True)
    backend_version = _require_str(backend_version, "backend_version", nonempty=True)
    family = capabilities.runner_family
    if family not in {"mrv1", "mrv2"}:
        raise BackendCompatibilityError(f"unsupported runner family {family!r}")
    if graph_mode not in {"eager", "piecewise"} and not (
        allow_legacy_mode and graph_mode == "full"
    ):
        raise BackendCompatibilityError(
            f"graph mode {graph_mode!r} is legacy-only; Phase 1a supports eager/piecewise"
        )
    capability_names = (
        f"graph_safety:{capabilities.graph_safety.name.lower()}",
        f"native_min_p:{bool(capabilities.native_min_p)}",
        f"tuple_sampler_return:{bool(capabilities.tuple_sampler_return)}",
        f"split_execute_and_sample:{bool(capabilities.split_execute_and_sample)}",
        *(f"sampler_stage:{stage.name.lower()}" for stage in sorted(capabilities.sampler_stages, key=lambda item: item.name)),
        *sorted(capabilities.capability_names()),
    )
    return BackendSnapshot(
        backend_id=capabilities.backend_id,
        runner_family=family,
        graph_mode=graph_mode,
        backend_version=backend_version,
        capabilities=capability_names,
    )

# These hooks are valid runtime placement points, but they are owned by the
# model/layer dispatcher rather than by a stable backend lifecycle anchor.
# A flow that reaches one of them remains a conservative legacy boundary.
_RUNTIME_NON_ANCHOR_HOOKS: frozenset[str] = frozenset(
    {
        "load_model.pre",
        "load_model.post",
        "layer.pre",
        "layer.post",
        "block.end",
        "stack.end",
    }
)


def backend_snapshot_from_capabilities(
    capabilities: BackendCapabilities,
    *,
    graph_mode: str,
    backend_version: str = "0.27.1",
) -> BackendSnapshot:
    """Lower typed backend capabilities without inspecting foreign objects."""
    mode = _require_str(graph_mode, "graph_mode", nonempty=True).lower()
    return _snapshot_from_capabilities(
        capabilities,
        graph_mode=mode,
        backend_version=backend_version,
        allow_legacy_mode=False,
    )


def backend_snapshot_from_resolution(resolution: BackendResolution) -> BackendSnapshot:
    """Lower an exact post-resolution backend context for the shadow compiler.

    FULL is represented as a legacy-only snapshot so existing dispatch keeps
    control while the WorkGraph report records the unsupported mode.
    """

    resolution_value = cast(object, resolution)
    if not isinstance(resolution_value, BackendResolution):
        raise TypeError("resolution must be BackendResolution")
    try:
        mode_name = _GRAPH_MODE_NAMES[resolution_value.graph_mode]
    except (AttributeError, KeyError) as exc:  # Defensive if GraphMode is extended without a lowering.
        raise BackendCompatibilityError(
            f"unsupported resolved graph mode {resolution_value.graph_mode!r}"
        ) from exc
    snapshot = _snapshot_from_capabilities(
        resolution_value.capabilities,
        graph_mode=mode_name,
        backend_version=resolution_value.backend_version,
        allow_legacy_mode=True,
    )
    return replace(snapshot, tp_size=resolution_value.tp_size)


def validate_backend_snapshot(backend: BackendSnapshot) -> None:
    """Validate a typed snapshot before lowering starts.

    ``BackendSnapshot`` itself enforces the structural descriptor contract at
    construction time. This function only checks the closed Phase 1a support
    matrix; it does not probe or coerce foreign runtime objects.
    """

    backend_value = cast(object, backend)
    if not isinstance(backend_value, BackendSnapshot):
        raise TypeError("backend must be BackendSnapshot")
    if backend_value.runner_family not in {"mrv1", "mrv2", "unresolved"}:
        raise BackendCompatibilityError(
            f"unsupported runner family {backend_value.runner_family!r}"
        )
    if not backend_value.backend_version.startswith("0.27."):
        raise BackendCompatibilityError(
            f"unsupported backend version family {backend_value.backend_version!r}"
        )
    if backend_value.graph_mode not in {"eager", "piecewise", "full"}:
        raise BackendCompatibilityError(f"unsupported graph mode {backend_value.graph_mode!r}")


def shadow_backend_snapshot() -> BackendSnapshot:
    """Describe startup metadata before a concrete runner is selected."""
    return BackendSnapshot(
        backend_id="vllm-0.27-shadow",
        runner_family="unresolved",
        graph_mode="eager",
        backend_version="0.27.1",
        capabilities=("shadow:true",),
    )


def anchor_desc(backend: BackendSnapshot, hook: str) -> AnchorDesc:
    """Create one fixed logical anchor from a validated backend snapshot."""
    validate_backend_snapshot(backend)
    logical_hook = _require_str(hook, "hook", nonempty=True)
    known = set(_LOGICAL_HOOKS)
    if logical_hook == "sampler.pre_draw":
        known.add(logical_hook)
    if logical_hook not in known:
        raise BackendCompatibilityError(f"unsupported logical hook {logical_hook!r}")
    return AnchorDesc(
        backend=backend.backend_id,
        runner=backend.runner_family,
        graph_mode=backend.graph_mode,
        hook=logical_hook,
    )


def available_logical_hooks(capabilities: BackendCapabilities) -> tuple[str, ...]:
    _validate_capabilities(capabilities)
    hooks: list[str] = list(_LOGICAL_HOOKS)
    if SamplerStage.BEFORE_DRAW in capabilities.sampler_stages:
        hooks.append("sampler.pre_draw")
    return tuple(hooks)


def is_runtime_non_anchor_hook(hook: str) -> bool:
    """Return whether a known runtime hook is intentionally not an anchor."""

    return _require_str(hook, "hook", nonempty=True) in _RUNTIME_NON_ANCHOR_HOOKS


Anchor = AnchorDesc

__all__ = [
    "Anchor",
    "AnchorDesc",
    "BackendCompatibilityError",
    "BackendResolution",
    "BackendSnapshot",
    "anchor_desc",
    "available_logical_hooks",
    "backend_snapshot_from_capabilities",
    "backend_snapshot_from_resolution",
    "is_runtime_non_anchor_hook",
    "shadow_backend_snapshot",
    "validate_backend_snapshot",
]
