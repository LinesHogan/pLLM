"""Canonical FTTT workload and typed host-service boundary."""

from tllm.canonical.workloads.fttt.host_service import FTTTFeedbackService
from tllm.canonical.workloads.fttt.judge import (
    FTTTDefaultJudge,
    FTTTJudgeService,
)
from tllm.canonical.workloads.fttt.types import (
    FTTTBarrierState,
    FTTTClosedLoopIntent,
    FTTTCompletion,
    FTTTCompletionBatch,
    FTTTConfig,
    FTTTControlReason,
    FTTTCpuReadiness,
    FTTTCudaReadiness,
    FTTTFinishReason,
    FTTTHostReadiness,
    FTTTHostTransaction,
    FTTTInstanceKey,
    FTTTInstanceState,
    FTTTJudgeResult,
    FTTTRetryAction,
    FTTTSpecialization,
)
from tllm.canonical.workloads.fttt.workload import FTTTWorkload

__all__ = [
    "FTTTBarrierState",
    "FTTTClosedLoopIntent",
    "FTTTCompletion",
    "FTTTCompletionBatch",
    "FTTTConfig",
    "FTTTControlReason",
    "FTTTCpuReadiness",
    "FTTTCudaReadiness",
    "FTTTDefaultJudge",
    "FTTTFeedbackService",
    "FTTTFinishReason",
    "FTTTHostReadiness",
    "FTTTHostTransaction",
    "FTTTInstanceKey",
    "FTTTInstanceState",
    "FTTTJudgeResult",
    "FTTTJudgeService",
    "FTTTRetryAction",
    "FTTTSpecialization",
    "FTTTWorkload",
]
