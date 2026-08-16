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
from typing import Any, Protocol


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
class BackendCapabilities:
    backend_id: str
    runner_family: str  # "mrv1" | "mrv2"
    sampler_stages: frozenset[SamplerStage]
    native_min_p: bool
    tuple_sampler_return: bool
    split_execute_and_sample: bool
    graph_safety: GraphSafety = GraphSafety.EAGER_ONLY


@dataclass
class RunnerBinding:
    """Per-runner-instance binding record."""

    driver: "BackendDriver"
    runner: Any
    capabilities: BackendCapabilities
    orig_methods: dict[str, Any] = field(default_factory=dict)
    active: bool = True


class BackendDriver(Protocol):
    """Owns engine-specific hook installation for one runner family."""

    backend_id: str

    def supports(self, info: Any) -> bool:
        """Structural check: can this driver drive the probed engine?"""
        ...

    def capabilities(self) -> BackendCapabilities:
        ...

    def install(self, *, core: Any, info: Any) -> None:
        """Install class-level hooks exactly once."""
        ...

    def unbind(self, binding: RunnerBinding) -> None:
        ...


class UnsupportedBackendError(RuntimeError):
    """Raised when no driver recognises the installed engine structure."""
