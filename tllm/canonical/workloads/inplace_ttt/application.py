"""Public typed composition for canonical dense fast weights."""

from __future__ import annotations

import torch

from tllm.canonical.host_runtime import CanonicalHostRuntimeFacts
from tllm.canonical.ptts.dense_page_planning import (
    DensePageCpuProviderFacts,
    DensePageCudaProviderFacts,
)
from tllm.canonical.workloads.inplace_ttt.cuda_execution import (
    InPlaceTTTCudaExecution,
)
from tllm.canonical.workloads.inplace_ttt.execution import InPlaceTTTExecution
from tllm.canonical.workloads.inplace_ttt.types import (
    CudaInPlaceTTTStepInput,
    InPlaceTTTCleanupReceipt,
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
    InPlaceTTTDiagnostics,
    InPlaceTTTSpecialization,
    InPlaceTTTStepInput,
    InPlaceTTTCudaStepHandle,
    InPlaceTTTStepResult,
)
from tllm.canonical.workloads.inplace_ttt.workload import (
    InPlaceTTTCompilation,
    InPlaceTTTWorkload,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.state_ref import StateRef
from tllm.contracts.state_representation import StateDeviceKind


class InPlaceTTTApplication:
    """Own one signed workload, PTTS provider and request directory."""

    __slots__ = ("_compilation", "_workload")

    def __init__(
        self,
        workload: InPlaceTTTWorkload,
        compilation: InPlaceTTTCompilation,
    ) -> None:
        if compilation.execution.config != workload.config:
            raise ValueError("In-Place-TTT application artifacts disagree")
        self._workload = workload
        self._compilation = compilation

    def admit(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        match self._compilation.execution:
            case InPlaceTTTExecution() as execution:
                return execution.admit(identities)
            case InPlaceTTTCudaExecution():
                raise RuntimeError("CUDA application admission requires a stream")

    def admit_cuda(
        self,
        identities: tuple[RequestIdentity, ...],
        stream: torch.cuda.Stream,
    ) -> tuple[StateRef, ...]:
        match self._compilation.execution:
            case InPlaceTTTCudaExecution() as execution:
                return execution.admit(identities, stream)
            case InPlaceTTTExecution():
                raise RuntimeError("CPU application does not accept CUDA admission")

    def step(self, step: InPlaceTTTStepInput) -> InPlaceTTTStepResult:
        match self._compilation.execution:
            case InPlaceTTTExecution() as execution:
                return execution.step(step)
            case InPlaceTTTCudaExecution():
                raise RuntimeError("CUDA application uses staged step methods")

    def begin_cuda_step(
        self,
        identity: RequestIdentity,
        stream: torch.cuda.Stream,
    ) -> InPlaceTTTCudaStepHandle:
        match self._compilation.execution:
            case InPlaceTTTCudaExecution() as execution:
                return execution.begin_step(identity, stream)
            case InPlaceTTTExecution():
                raise RuntimeError("CPU application has no CUDA step")

    def submit_cuda(
        self,
        handle: InPlaceTTTCudaStepHandle,
        step: InPlaceTTTStepInput,
        producer_stream: torch.cuda.Stream,
    ) -> None:
        match self._compilation.execution:
            case InPlaceTTTCudaExecution() as execution:
                execution.submit_captured(
                    handle,
                    CudaInPlaceTTTStepInput.assert_produced_on(
                        step,
                        producer_stream,
                    ),
                )
            case InPlaceTTTExecution():
                raise RuntimeError("CPU application does not accept CUDA operands")

    def complete_cuda_sampler(
        self,
        handle: InPlaceTTTCudaStepHandle,
        sampler_stream: torch.cuda.Stream,
    ) -> None:
        match self._compilation.execution:
            case InPlaceTTTCudaExecution() as execution:
                execution.complete_sampler(handle, sampler_stream)
            case InPlaceTTTExecution():
                raise RuntimeError("CPU application has no CUDA sampler")

    def finish_cuda_step(
        self,
        handle: InPlaceTTTCudaStepHandle,
    ) -> InPlaceTTTStepResult:
        match self._compilation.execution:
            case InPlaceTTTCudaExecution() as execution:
                return execution.finish_step(handle)
            case InPlaceTTTExecution():
                raise RuntimeError("CPU application has no CUDA step")

    def finish(
        self,
        identities: tuple[RequestIdentity, ...],
    ) -> InPlaceTTTCleanupReceipt:
        return self._compilation.execution.retire(identities)

    def diagnostics(self) -> InPlaceTTTDiagnostics:
        return self._compilation.execution.diagnostics()


def create_inplace_ttt_application(
    config: InPlaceTTTConfig,
    specialization: InPlaceTTTSpecialization,
    *,
    max_admitted_lineages: int,
    state_memory_budget_bytes: int,
    plan_epoch: int = 1,
) -> InPlaceTTTApplication:
    facts = CanonicalHostRuntimeFacts(specialization.device, plan_epoch)
    match specialization.device_kind:
        case StateDeviceKind.CPU:
            providers: tuple[
                DensePageCpuProviderFacts | DensePageCudaProviderFacts,
                ...,
            ] = (DensePageCpuProviderFacts("canonical.cpu.full_root@1", 2),)
            extra_capabilities = ("transition/full_root@1",)
        case StateDeviceKind.CUDA:
            providers = (
                DensePageCudaProviderFacts(
                    "canonical.cuda.full_root@1",
                    specialization.device_ordinal,
                    3,
                ),
            )
            extra_capabilities = (
                "state/fenced_reclaim@1",
                "transition/full_root@1",
            )
    workload = InPlaceTTTWorkload(config)
    compilation = workload.compile(
        specialization,
        InPlaceTTTCompileOptions(
            facts.timeline(extra_capabilities),
            providers,
            max_admitted_lineages,
            state_memory_budget_bytes,
            plan_epoch,
        ),
    )
    return InPlaceTTTApplication(workload, compilation)


__all__ = ["InPlaceTTTApplication", "create_inplace_ttt_application"]
