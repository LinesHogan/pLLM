"""Single typed backend-adapter seam for compiled workloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, Self

from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.workload_bindings import BackendRuntimeBinding
from tllm.workload.config import WorkloadConfig


class WorkloadBackendAdapter(Protocol):
    @property
    def requires_specialization(self) -> bool: ...

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
    @property
    def requires_specialization(self) -> bool:
        return False

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
        return BackendRuntimeBinding()


__all__ = [
    "BackendRuntimeBinding",
    "NoBackendAdapter",
    "WorkloadBackendAdapter",
]
