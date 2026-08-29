"""Closed, immutable process-local bindings for a native workload.

Signed compilation data deliberately contains no live Python handles.  This
module is the one place where those handles are grouped after the compiler has
selected a native workload implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar

from tllm.contracts.consumer_stats import TrainingStats
from tllm.runtime.backend_region_binders import BackendRegionBinderSlot
from tllm.runtime.specialization import RuntimeSpecializationSpec

if TYPE_CHECKING:
    from tllm.ports.residual_stream import ResidualLocator
    from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider
    from tllm.runtime.workload_plugins import NativeWorkload, WorkloadWorkerSpec


InputProvider = Callable[[int], object]
OutputSink = Callable[[object, int], None]
StateSnapshot = Callable[[int], Mapping[Hashable, int]]


CallbackT = TypeVar("CallbackT", bound=Callable[..., object])


def _frozen_callbacks(bindings: Mapping[str, CallbackT]) -> Mapping[str, CallbackT]:
    if any(not name for name in bindings):
        raise ValueError("runtime binding names must be non-empty")
    return MappingProxyType(dict(bindings))


def _empty_input_providers() -> Mapping[str, InputProvider]:
    return MappingProxyType({})


def _empty_output_sinks() -> Mapping[str, OutputSink]:
    return MappingProxyType({})


def _empty_state_snapshots() -> Mapping[str, StateSnapshot]:
    return MappingProxyType({})


def _empty_backend_binders() -> Mapping[str, BackendRegionBinderSlot]:
    return MappingProxyType({})


def _empty_specialization_key() -> Mapping[str, str]:
    return MappingProxyType({})


def _empty_residual_paths() -> Mapping["ResidualLocator", str]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class RuntimePortRegistry:
    """The complete local endpoint registry for a compiled workload."""

    inputs: Mapping[str, InputProvider] = field(default_factory=_empty_input_providers)
    outputs: Mapping[str, OutputSink] = field(default_factory=_empty_output_sinks)

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _frozen_callbacks(self.inputs))
        object.__setattr__(self, "outputs", _frozen_callbacks(self.outputs))


@dataclass(frozen=True, slots=True)
class RuntimeStateRegistry:
    """Immutable state-version readers keyed by declaration-local state name."""

    snapshots: Mapping[str, StateSnapshot] = field(default_factory=_empty_state_snapshots)

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", _frozen_callbacks(self.snapshots))


@dataclass(frozen=True, slots=True)
class BackendRuntimeBinding:
    """One compiler-selected backend adapter binding for a workload."""

    ports: RuntimePortRegistry = field(default_factory=RuntimePortRegistry)
    binders: Mapping[str, BackendRegionBinderSlot] = field(
        default_factory=_empty_backend_binders
    )

    def __post_init__(self) -> None:
        if any(
            not implementation_id
            or slot.implementation_id != implementation_id
            for implementation_id, slot in self.binders.items()
        ):
            raise ValueError("backend binder IDs must match their binding slots")
        object.__setattr__(self, "binders", MappingProxyType(dict(self.binders)))

@dataclass(frozen=True, slots=True)
class NoRuntimeServices:
    """A declaration with no local lifecycle service."""


@dataclass(frozen=True, slots=True)
class DiagnosticsRuntimeServices:
    """A workload that publishes diagnostics but owns no lifecycle state."""

    diagnostics: Callable[[], Mapping[str, int | float]]

@dataclass(frozen=True, slots=True)
class NoPerRequestStats:
    """A lifecycle without per-request accounting."""


@dataclass(frozen=True, slots=True)
class PerRequestStats:
    read: Callable[[bool], Mapping[int, TrainingStats]]

PerRequestStatsService = NoPerRequestStats | PerRequestStats


@dataclass(frozen=True, slots=True)
class LocalRuntimeServices:
    """Bound lifecycle methods for one workload, without nullable slots."""

    cold_specialize: Callable[[RuntimeSpecializationSpec], None]
    diagnostics: Callable[[], Mapping[str, int | float]]
    stats: Callable[[bool], TrainingStats]
    set_enabled: Callable[[bool], None]
    is_enabled: Callable[[], bool]
    synchronize: Callable[[], None]
    per_request_stats: PerRequestStatsService = field(
        default_factory=NoPerRequestStats
    )

RuntimeServices = NoRuntimeServices | DiagnosticsRuntimeServices | LocalRuntimeServices


@dataclass(frozen=True, slots=True)
class NoSamplerBinding:
    """A workload without a sampler output declaration."""


@dataclass(frozen=True, slots=True)
class SamplerRuntimeBinding:
    provider: "SamplerModifierProvider"


SamplerBinding = NoSamplerBinding | SamplerRuntimeBinding


@dataclass(frozen=True, slots=True)
class NoWorkerBinding:
    """Local-only binding; an external control plane supplies its recipe."""


@dataclass(frozen=True, slots=True)
class WorkerRuntimeBinding:
    spec: "WorkloadWorkerSpec"


WorkerBinding = NoWorkerBinding | WorkerRuntimeBinding


@dataclass(frozen=True, slots=True)
class WorkloadRuntimeBinding:
    """Complete local binding for one native workload declaration.

    Absence is represented by a closed variant or an empty immutable registry,
    never by a nullable callback field.
    """

    owner: "NativeWorkload"
    ports: RuntimePortRegistry = field(default_factory=RuntimePortRegistry)
    states: RuntimeStateRegistry = field(default_factory=RuntimeStateRegistry)
    services: RuntimeServices = field(default_factory=NoRuntimeServices)
    sampler: SamplerBinding = field(default_factory=NoSamplerBinding)
    worker: WorkerBinding = field(default_factory=NoWorkerBinding)
    specialization_key: Mapping[str, str] = field(
        default_factory=_empty_specialization_key
    )
    residual_paths: Mapping["ResidualLocator", str] = field(
        default_factory=_empty_residual_paths
    )
    backend: BackendRuntimeBinding = field(default_factory=BackendRuntimeBinding)

    def __post_init__(self) -> None:
        if any(not name or not value for name, value in self.specialization_key.items()):
            raise ValueError("specialization keys must be non-empty strings")
        object.__setattr__(
            self,
            "specialization_key",
            MappingProxyType(dict(sorted(self.specialization_key.items()))),
        )
        object.__setattr__(self, "residual_paths", MappingProxyType(dict(self.residual_paths)))

__all__ = [
    "BackendRuntimeBinding",
    "DiagnosticsRuntimeServices",
    "InputProvider",
    "LocalRuntimeServices",
    "NoPerRequestStats",
    "NoRuntimeServices",
    "NoSamplerBinding",
    "NoWorkerBinding",
    "OutputSink",
    "PerRequestStats",
    "RuntimePortRegistry",
    "RuntimeStateRegistry",
    "SamplerRuntimeBinding",
    "StateSnapshot",
    "WorkerRuntimeBinding",
    "WorkloadRuntimeBinding",
]
