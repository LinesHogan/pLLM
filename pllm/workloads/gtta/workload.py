"""GTTA dataflow over the original CPU host transaction."""

from __future__ import annotations

from functools import partial
import pllm
from pllm.workloads.gtta.environment import GTTAEnvironmentService
from pllm.workloads.gtta import state as math
from pllm.workloads.gtta.types import GTTAConfig, GTTAHostRequest, GTTAHostTransaction, GTTAResult


class GTTAWorkload(pllm.HostWorkload[pllm.HostTransactionCoordinator[GTTAHostRequest, GTTAHostTransaction, GTTAResult]]):
    def __init__(self, environment: GTTAEnvironmentService | None = None) -> None:
        self.environment = environment if environment is not None else GTTAEnvironmentService(GTTAConfig())

    @property
    def config(self) -> GTTAConfig:
        return self.environment.config

    def declare_host(self, facts: pllm.HostRuntimeFacts) -> pllm.TransactionHostBinding[
        GTTAHostRequest, GTTAHostTransaction, math.GTTAStepContext, math.GTTADynamicsSnapshot, GTTAResult,
    ]:
        if facts.device.type != "cpu":
            raise ValueError("GTTA environment execution requires CPU placement")
        return pllm.transaction("gtta.dynamics", facts, config=self.config,
            state_factory=math.GTTADynamicsState, service=self.environment,
            update=partial(math.update, self.config), query=math.query, publish=math.publish)


__all__ = ["GTTAWorkload"]
