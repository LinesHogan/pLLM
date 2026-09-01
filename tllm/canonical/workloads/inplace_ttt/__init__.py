"""Canonical In-Place-TTT over request-scoped PTTS roots."""

from tllm.canonical.workloads.inplace_ttt.application import (
    InPlaceTTTApplication,
    create_inplace_ttt_application,
)
from tllm.canonical.workloads.inplace_ttt.execution import InPlaceTTTExecution
from tllm.canonical.workloads.inplace_ttt.cuda_execution import (
    InPlaceTTTCudaExecution,
)
from tllm.canonical.workloads.inplace_ttt.types import (
    CudaInPlaceTTTStepInput,
    InPlaceTTTCleanupReceipt,
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
    InPlaceTTTDiagnostics,
    InPlaceTTTCudaStepHandle,
    InPlaceTTTOperandPair,
    InPlaceTTTPhase,
    InPlaceTTTPinnedDelta,
    InPlaceTTTSpecialization,
    InPlaceTTTStepInput,
    InPlaceTTTStepResult,
    InPlaceTTTTargetResult,
    InPlaceTTTTargetSpecialization,
)
from tllm.canonical.workloads.inplace_ttt.workload import (
    InPlaceTTTCompilation,
    InPlaceTTTWorkload,
)

__all__ = [
    "CudaInPlaceTTTStepInput",
    "InPlaceTTTApplication",
    "InPlaceTTTCleanupReceipt",
    "InPlaceTTTCompilation",
    "InPlaceTTTCompileOptions",
    "InPlaceTTTConfig",
    "InPlaceTTTDiagnostics",
    "InPlaceTTTCudaExecution",
    "InPlaceTTTCudaStepHandle",
    "InPlaceTTTExecution",
    "InPlaceTTTOperandPair",
    "InPlaceTTTPhase",
    "InPlaceTTTPinnedDelta",
    "InPlaceTTTSpecialization",
    "InPlaceTTTStepInput",
    "InPlaceTTTStepResult",
    "InPlaceTTTTargetResult",
    "InPlaceTTTTargetSpecialization",
    "InPlaceTTTWorkload",
    "create_inplace_ttt_application",
]
