"""Public typed composition for canonical GTTA environment interaction."""

from __future__ import annotations

import torch

from tllm.canonical.host_runtime import CanonicalHostRuntimeFacts
from tllm.canonical.workloads.gtta.environment import GTTAEnvironmentService
from tllm.canonical.workloads.gtta.execution import (
    GTTADiagnostics,
    GTTAExecutionResult,
)
from tllm.canonical.workloads.gtta.session import GTTASession
from tllm.canonical.workloads.gtta.types import (
    GTTAConfig,
    GTTAQuery,
    TrajectoryFeedback,
)
from tllm.canonical.workloads.gtta.workload import (
    GTTACompilation,
    GTTACompileOptions,
    GTTAWorkload,
)


class GTTAApplication:
    """Own the environment service and its instance-scoped signed graph."""

    __slots__ = ("_compilation", "_environment", "_session", "_workload")

    def __init__(
        self,
        workload: GTTAWorkload,
        compilation: GTTACompilation,
    ) -> None:
        if compilation.execution.config != workload.config:
            raise ValueError("GTTA application requires exact shared artifacts")
        environment = GTTAEnvironmentService(workload.config)
        self._workload = workload
        self._compilation = compilation
        self._environment = environment
        self._session = GTTASession(environment, compilation.execution)

    def observe(self, feedback: TrajectoryFeedback) -> bool:
        return self._environment.enqueue_feedback(feedback)

    def query(self, query: GTTAQuery, *, limit: int = 0) -> GTTAExecutionResult:
        return self._session.query(query, limit)

    def diagnostics(self) -> GTTADiagnostics:
        return self._compilation.execution.diagnostics()

    @property
    def queued_trajectories(self) -> int:
        return self._environment.queued_trajectories

    @property
    def dropped_trajectories(self) -> int:
        return self._environment.dropped_trajectories


def create_gtta_application(
    config: GTTAConfig = GTTAConfig(),
    *,
    plan_epoch: int = 1,
) -> GTTAApplication:
    runtime_facts = CanonicalHostRuntimeFacts(torch.device("cpu"), plan_epoch)
    workload = GTTAWorkload(config)
    compilation = workload.compile(
        GTTACompileOptions(runtime_facts.timeline(), plan_epoch)
    )
    return GTTAApplication(workload, compilation)


__all__ = ["GTTAApplication", "create_gtta_application"]
