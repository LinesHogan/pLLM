"""Closed, immutable process-local bindings for a native workload.

Signed compilation data deliberately contains no live Python handles.  This
module is the one place where those handles are grouped after the compiler has
selected a native workload implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from tllm.contracts.consumer_stats import TrainingStats
from tllm.runtime.backend_region_binders import BackendRegionBinderSlot
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.parameter_overlay import OverlayChannel, OverlayVersionStore
from tllm.runtime.request_trainable_state import RequestStateRuntimeService

if TYPE_CHECKING:
    import torch
    from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
    from tllm.ports.residual_stream import ResidualLocator
    from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider
    from tllm.runtime.workload_plugins import NativeWorkload, WorkloadWorkerSpec


StateSnapshot = Callable[[int], Mapping[Hashable, int]]
InputValueT_co = TypeVar("InputValueT_co", covariant=True)
OutputValueT_contra = TypeVar("OutputValueT_contra", contravariant=True)


@dataclass(frozen=True, slots=True)
class RuntimeInputBinding(Generic[InputValueT_co]):
    provider: Callable[[int], InputValueT_co]

    def read(self, step_id: int) -> InputValueT_co:
        return self.provider(step_id)


@dataclass(frozen=True, slots=True)
class RuntimeOutputBinding(Generic[OutputValueT_contra]):
    sink: Callable[[OutputValueT_contra, int], None]

    def publish(self, value: OutputValueT_contra, step_id: int) -> None:
        self.sink(value, step_id)


BindingT = TypeVar("BindingT")


def _frozen_bindings(bindings: Mapping[str, BindingT]) -> Mapping[str, BindingT]:
    if any(not name for name in bindings):
        raise ValueError("runtime binding names must be non-empty")
    return MappingProxyType(dict(bindings))


def _empty_input_providers() -> Mapping[str, RuntimeInputBinding[object]]:
    return MappingProxyType({})


def _empty_output_sinks() -> Mapping[str, RuntimeOutputBinding[object]]:
    return MappingProxyType({})


def _empty_state_snapshots() -> Mapping[str, StateSnapshot]:
    return MappingProxyType({})


def _empty_backend_binders() -> Mapping[str, BackendRegionBinderSlot]:
    return MappingProxyType({})


def _empty_specialization_key() -> Mapping[str, str]:
    return MappingProxyType({})


def _empty_residual_paths() -> Mapping["ResidualLocator", str]:
    return MappingProxyType({})


def _empty_parameter_overlays() -> Mapping[OverlayChannel, OverlayVersionStore]:
    return MappingProxyType({})


class OptimizerTransitionService(Protocol):
    """Typed split between capturable device work and one host commit."""

    def device_step(self, loss: "torch.Tensor") -> None: ...

    def commit(self, loss: "torch.Tensor") -> None: ...


def _empty_optimizer_transitions() -> Mapping[str, OptimizerTransitionService]:
    return MappingProxyType({})


def _empty_request_state_services(
) -> Mapping[str, RequestStateRuntimeService["torch.nn.Module"]]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class RuntimePortRegistry:
    """The complete local endpoint registry for a compiled workload."""

    inputs: Mapping[str, RuntimeInputBinding[object]] = field(
        default_factory=_empty_input_providers
    )
    outputs: Mapping[str, RuntimeOutputBinding[object]] = field(
        default_factory=_empty_output_sinks
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", _frozen_bindings(self.inputs))
        object.__setattr__(self, "outputs", _frozen_bindings(self.outputs))


@dataclass(frozen=True, slots=True)
class RuntimeStateRegistry:
    """Immutable state-version readers keyed by declaration-local state name."""

    snapshots: Mapping[str, StateSnapshot] = field(default_factory=_empty_state_snapshots)

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshots", _frozen_bindings(self.snapshots))


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
    stage_identity: Callable[
        [
            int,
            tuple[str, ...],
            tuple[int, ...],
            "torch.Tensor | None",
        ],
        None,
    ]


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
    packed_hidden_resources: tuple["PackedHiddenResourceBinding", ...] = ()
    backend: BackendRuntimeBinding = field(default_factory=BackendRuntimeBinding)
    parameter_overlays: Mapping[OverlayChannel, OverlayVersionStore] = field(
        default_factory=_empty_parameter_overlays
    )
    optimizer_transitions: Mapping[str, OptimizerTransitionService] = field(
        default_factory=_empty_optimizer_transitions
    )
    request_state_services: Mapping[
        str, RequestStateRuntimeService["torch.nn.Module"]
    ] = field(
        default_factory=_empty_request_state_services
    )

    def __post_init__(self) -> None:
        if any(not name or not value for name, value in self.specialization_key.items()):
            raise ValueError("specialization keys must be non-empty strings")
        object.__setattr__(
            self,
            "specialization_key",
            MappingProxyType(dict(sorted(self.specialization_key.items()))),
        )
        object.__setattr__(self, "residual_paths", MappingProxyType(dict(self.residual_paths)))
        packed_locator_keys = tuple(
            binding.locator_key for binding in self.packed_hidden_resources
        )
        if tuple(sorted(packed_locator_keys)) != packed_locator_keys or len(
            set(packed_locator_keys)
        ) != len(packed_locator_keys):
            raise ValueError(
                "packed hidden resources must have sorted unique locators"
            )
        object.__setattr__(
            self,
            "parameter_overlays",
            MappingProxyType(dict(self.parameter_overlays)),
        )
        object.__setattr__(
            self,
            "optimizer_transitions",
            _frozen_bindings(self.optimizer_transitions),
        )
        if any(
            service.resource != resource
            for resource, service in self.request_state_services.items()
        ):
            raise ValueError(
                "request-state service resource names must match their registry keys"
            )
        object.__setattr__(
            self,
            "request_state_services",
            _frozen_bindings(self.request_state_services),
        )

__all__ = [
    "BackendRuntimeBinding",
    "DiagnosticsRuntimeServices",
    "LocalRuntimeServices",
    "NoPerRequestStats",
    "NoRuntimeServices",
    "NoSamplerBinding",
    "NoWorkerBinding",
    "OptimizerTransitionService",
    "PerRequestStats",
    "RuntimePortRegistry",
    "RuntimeInputBinding",
    "RuntimeOutputBinding",
    "RuntimeStateRegistry",
    "RequestStateRuntimeService",
    "SamplerRuntimeBinding",
    "StateSnapshot",
    "WorkerRuntimeBinding",
    "WorkloadRuntimeBinding",
]
