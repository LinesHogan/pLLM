"""GTTA on the host: fold queued environment feedback into the dynamics memory, then rank a query's actions."""

from __future__ import annotations

from pllm.workloads.gtta.environment import GTTAEnvironmentService
from pllm.workloads.gtta.state import GTTADynamicsState
from pllm.workloads.gtta.types import GTTAConfig, GTTAHostRequest, GTTAResult


class GTTAWorkload:
    """One environment queue and the dynamics memory its events update."""

    def __init__(self, environment: GTTAEnvironmentService | None = None) -> None:
        self.environment = environment if environment is not None else GTTAEnvironmentService(GTTAConfig())
        self.dynamics = GTTADynamicsState()
        self._runs = 0

    @property
    def config(self) -> GTTAConfig:
        return self.environment.config

    def run(self, request: GTTAHostRequest) -> GTTAResult:
        """Update the memory with up to `request.limit` queued events, then answer `request.query`. A failure restores
        the memory and requeues the events."""
        self._runs += 1
        trajectories = self.environment.take(request.limit)
        snapshot = self.dynamics.snapshot()
        try:
            update = self.dynamics.update(trajectories, self.config)
            return GTTAResult(self._runs, update, self.dynamics.query(request.query))
        except BaseException:
            self.dynamics.restore(snapshot)
            self.environment.requeue(trajectories)
            raise


__all__ = ["GTTAWorkload"]
