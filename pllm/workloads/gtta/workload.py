"""GTTA on the host: fold queued environment feedback into the dynamics memory, then rank a query's actions."""

from __future__ import annotations

from pllm.workloads.gtta.environment import GTTAEnvironment
from pllm.workloads.gtta.state import GTTADynamicsState
from pllm.workloads.gtta.types import GTTAConfig, GTTAHostRequest, GTTAResult


class GTTAWorkload:
    """One environment queue and the dynamics memory its events update."""

    def __init__(self, environment: GTTAEnvironment | None = None) -> None:
        self.environment = environment if environment is not None else GTTAEnvironment(GTTAConfig())
        self.dynamics = GTTADynamicsState()

    @property
    def config(self) -> GTTAConfig:
        return self.environment.config

    def run(self, request: GTTAHostRequest) -> GTTAResult:
        """Consume queued feedback, update the host memory, then answer the query.

        Applied feedback remains in memory if the query fails.
        """
        trajectories = self.environment.take(request.limit)
        self.dynamics.update(trajectories, self.config)
        return GTTAResult(len(trajectories), self.dynamics.query(request.query))


__all__ = ["GTTAWorkload"]
