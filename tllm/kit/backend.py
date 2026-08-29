"""Single typed backend-adapter seam for compiled workloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, Self

from tllm.runtime.backend_region_binders import BackendRegionBinderSlot
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.workload.config import WorkloadConfig


@dataclass(frozen=True, slots=True)
class BackendRuntimeBinding:
    provided_inputs: Mapping[str, Callable[[int], object]]
    binders: Mapping[str, BackendRegionBinderSlot]


class WorkloadBackendAdapter(Protocol):
    @classmethod
    def create(cls, config: WorkloadConfig) -> Self: ...

    @classmethod
    def reconstruct(
        cls, config: WorkloadConfig, wire: Mapping[str, object]
    ) -> Self: ...

    def specialize(self, spec: RuntimeSpecializationSpec) -> None: ...

    def specialization_key(self) -> Mapping[str, str]: ...

    def worker_wire(self) -> Mapping[str, object]: ...

    def runtime_binding(self) -> BackendRuntimeBinding: ...


class NoBackendAdapter:
    @classmethod
    def create(cls, config: WorkloadConfig) -> Self:
        del config
        return cls()

    @classmethod
    def reconstruct(
        cls, config: WorkloadConfig, wire: Mapping[str, object]
    ) -> Self:
        del config, wire
        return cls()

    def specialize(self, spec: RuntimeSpecializationSpec) -> None:
        del spec

    def specialization_key(self) -> Mapping[str, str]:
        return {}

    def worker_wire(self) -> Mapping[str, object]:
        return {}

    def runtime_binding(self) -> BackendRuntimeBinding:
        return BackendRuntimeBinding({}, {})


__all__ = [
    "BackendRuntimeBinding",
    "NoBackendAdapter",
    "WorkloadBackendAdapter",
]
