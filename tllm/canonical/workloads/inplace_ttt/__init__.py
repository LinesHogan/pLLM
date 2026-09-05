"""Canonical In-Place-TTT over request-scoped PTTS roots."""

from tllm.canonical.workloads.inplace_ttt.types import (
    CudaInPlaceTTTStepBatchInput,
    InPlaceTTTCleanupReceipt,
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
    InPlaceTTTDiagnostics,
    InPlaceTTTStepHandle,
    InPlaceTTTOperandPair,
    InPlaceTTTPhase,
    InPlaceTTTSpecialization,
    InPlaceTTTStepInput,
    InPlaceTTTStepBatch,
    InPlaceTTTStepBatchResult,
    InPlaceTTTStepResult,
    InPlaceTTTTargetResult,
    InPlaceTTTTargetSpecialization,
)
from tllm.canonical.workloads.inplace_ttt.workload import (
    CanonicalDenseOverlayExecution,
    InPlaceTTTCompilation,
    InPlaceTTTWorkload,
)
__all__ = [
    "CanonicalDenseOverlayExecution",
    "CudaInPlaceTTTStepBatchInput",
    "InPlaceTTTCleanupReceipt",
    "InPlaceTTTCompilation",
    "InPlaceTTTCompileOptions",
    "InPlaceTTTConfig",
    "InPlaceTTTDiagnostics",
    "InPlaceTTTStepHandle",
    "InPlaceTTTOperandPair",
    "InPlaceTTTPhase",
    "InPlaceTTTSpecialization",
    "InPlaceTTTStepInput",
    "InPlaceTTTStepBatch",
    "InPlaceTTTStepBatchResult",
    "InPlaceTTTStepResult",
    "InPlaceTTTTargetResult",
    "InPlaceTTTTargetSpecialization",
    "InPlaceTTTWorkload",
]
