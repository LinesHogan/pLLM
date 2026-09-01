"""Public typed composition for canonical dense fast weights."""

from __future__ import annotations

import torch

from tllm.canonical.host_runtime import CanonicalHostRuntimeFacts
from tllm.canonical.ptts.dense_page_planning import DensePageCpuProviderFacts
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTCleanupReceipt,
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
    InPlaceTTTDiagnostics,
    InPlaceTTTSpecialization,
    InPlaceTTTStepInput,
    InPlaceTTTStepResult,
)
from tllm.canonical.workloads.inplace_ttt.workload import (
    InPlaceTTTCompilation,
    InPlaceTTTWorkload,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.state_ref import StateRef


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
        return self._compilation.execution.admit(identities)

    def step(self, step: InPlaceTTTStepInput) -> InPlaceTTTStepResult:
        return self._compilation.execution.step(step)

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
    facts = CanonicalHostRuntimeFacts(torch.device("cpu"), plan_epoch)
    workload = InPlaceTTTWorkload(config)
    compilation = workload.compile(
        specialization,
        InPlaceTTTCompileOptions(
            facts.timeline(("transition/full_root@1",)),
            (DensePageCpuProviderFacts("canonical.cpu.full_root@1", 2),),
            max_admitted_lineages,
            state_memory_budget_bytes,
            plan_epoch,
        ),
    )
    return InPlaceTTTApplication(workload, compilation)


__all__ = ["InPlaceTTTApplication", "create_inplace_ttt_application"]
