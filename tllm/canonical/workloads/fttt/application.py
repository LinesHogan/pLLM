"""Public typed composition for canonical FTTT host submissions."""

from __future__ import annotations

import torch

from tllm.canonical.host_runtime import CanonicalHostRuntimeFacts
from tllm.canonical.workloads.fttt.execution import (
    FTTTDiagnostics,
    FTTTExecutionResult,
)
from tllm.canonical.workloads.fttt.host_service import FTTTFeedbackService
from tllm.canonical.workloads.fttt.judge import FTTTJudgeService
from tllm.canonical.workloads.fttt.session import FTTTExecuted, FTTTSession
from tllm.canonical.workloads.fttt.types import (
    FTTTCompletionBatch,
    FTTTConfig,
    FTTTInstanceKey,
    FTTTInstanceState,
    FTTTSpecialization,
)
from tllm.canonical.workloads.fttt.workload import (
    FTTTCompilation,
    FTTTCompileOptions,
    FTTTWorkload,
)


class FTTTApplication:
    """Own the workload, judge service, signed execution, and atomic session."""

    __slots__ = ("_compilation", "_service", "_session", "_workload")

    def __init__(
        self,
        workload: FTTTWorkload,
        compilation: FTTTCompilation,
        judge: FTTTJudgeService,
    ) -> None:
        if (
            type(workload) is not FTTTWorkload
            or type(compilation) is not FTTTCompilation
            or compilation.execution.config != workload.config
        ):
            raise TypeError("FTTT application requires exact shared artifacts")
        service = FTTTFeedbackService(workload.config, judge)
        session = FTTTSession(service, compilation.execution)
        self._workload = workload
        self._compilation = compilation
        self._service = service
        self._session = session

    def submit_cpu(
        self,
        completions: FTTTCompletionBatch,
        hidden_rows: torch.Tensor,
    ) -> FTTTExecutionResult:
        self._service.enqueue(completions, hidden_rows)
        return self._run_submitted()

    def submit_cuda(
        self,
        completions: FTTTCompletionBatch,
        hidden_rows: torch.Tensor,
        producer_stream: torch.cuda.Stream,
    ) -> FTTTExecutionResult:
        self._service.enqueue_cuda(completions, hidden_rows, producer_stream)
        return self._run_submitted()

    def diagnostics(self) -> FTTTDiagnostics:
        return self._compilation.execution.diagnostics()

    def instance_state(self, key: FTTTInstanceKey) -> FTTTInstanceState:
        return self._service.instance_state(key)

    def _run_submitted(self) -> FTTTExecutionResult:
        polled = self._session.run_next()
        match polled:
            case FTTTExecuted(result=result):
                return result
            case _:
                raise RuntimeError("FTTT submitted transaction was not executed")


def create_fttt_application(
    config: FTTTConfig,
    specialization: FTTTSpecialization,
    judge: FTTTJudgeService,
    *,
    plan_epoch: int = 1,
) -> FTTTApplication:
    if (
        type(config) is not FTTTConfig
        or type(specialization) is not FTTTSpecialization
        or type(plan_epoch) is not int
        or plan_epoch < 1
    ):
        raise TypeError("FTTT application factory requires exact cold facts")
    runtime_facts = CanonicalHostRuntimeFacts(
        specialization.device,
        plan_epoch,
    )
    workload = FTTTWorkload(config)
    compilation = workload.compile(
        specialization,
        FTTTCompileOptions(runtime_facts.timeline(), plan_epoch),
    )
    return FTTTApplication(workload, compilation, judge)


__all__ = ["FTTTApplication", "create_fttt_application"]
