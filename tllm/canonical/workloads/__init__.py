"""Workloads implemented only on the clean-room canonical core."""

from tllm.canonical.workloads.base import CanonicalWorkload
from tllm.canonical.workloads.dummy import (
    DummyConfig,
    DummyExported,
    DummySkipped,
    DummyStep,
    DummyWorkload,
)
from tllm.canonical.workloads.esamp_request import (
    ESampRequestCompileOptions,
    ESampRequestConfig,
    ESampRequestCpuExecution,
    ESampRequestCpuStepHandle,
    ESampRequestCudaExecution,
    ESampRequestCudaStepHandle,
    ESampRequestExecution,
    ESampRequestSpecialization,
    ESampRequestStepHandle,
    ESampRequestWorkload,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestDiagnostics,
    ESampRequestPackedHiddenBindings,
    ESampRequestStepInput,
    ESampRequestStepResult,
)

__all__ = [
    "CanonicalWorkload",
    "DummyConfig",
    "DummyExported",
    "DummySkipped",
    "DummyStep",
    "DummyWorkload",
    "ESampRequestCompileOptions",
    "ESampRequestConfig",
    "ESampRequestCpuExecution",
    "ESampRequestCpuStepHandle",
    "ESampRequestCudaExecution",
    "ESampRequestCudaStepHandle",
    "ESampRequestDiagnostics",
    "ESampRequestExecution",
    "ESampRequestPackedHiddenBindings",
    "ESampRequestSpecialization",
    "ESampRequestStepInput",
    "ESampRequestStepHandle",
    "ESampRequestStepResult",
    "ESampRequestWorkload",
]
