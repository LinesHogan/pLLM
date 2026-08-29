"""Strict-only public API composition fixture for migrated native workloads."""

from __future__ import annotations

import torch

from tllm.consumers.dummy.workload import DummyConfig, DummyWorkload
from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload
from tllm.consumers.override.workload import OverRIDEConfig, OverRIDEWorkload
from tllm.consumers.recirculation.workload import (
    RecirculationConfig,
    RecirculationWorkload,
)
from tllm.consumers.reft.workload import ReftConfig, ReftParameters, ReftWorkload
from tllm.consumers.ttrl.rewards import TTRLClosedLoopIntent
from tllm.consumers.ttrl.workload import TTRLConfig, TTRLWorkload
from tllm.runtime.runtime import TllmRuntime

NativeWorkloadSet = (
    DummyWorkload
    | ESampWorkload
    | OverRIDEWorkload
    | ReftWorkload
    | TTRLWorkload
    | RecirculationWorkload
)


def declarations() -> tuple[
    DummyWorkload,
    ESampWorkload,
    OverRIDEWorkload,
    ReftWorkload,
    TTRLWorkload,
    RecirculationWorkload,
]:
    return (
        DummyWorkload(DummyConfig()),
        ESampWorkload(ESampConfig()),
        OverRIDEWorkload(OverRIDEConfig()),
        ReftWorkload(ReftConfig()),
        TTRLWorkload(TTRLConfig()),
        RecirculationWorkload(RecirculationConfig()),
    )


def runtime_composition() -> TllmRuntime[NativeWorkloadSet]:
    """Keep workload construction and runtime admission in one strict fixture."""

    runtime = TllmRuntime(workloads=declarations())
    typed_workloads: tuple[NativeWorkloadSet, ...] = runtime.workloads
    del typed_workloads
    return runtime


def typed_ports(
    runtime: TllmRuntime[TTRLWorkload],
    workload: TTRLWorkload,
    intent: TTRLClosedLoopIntent,
) -> None:
    hidden = torch.zeros(2, 8)
    rewards = torch.zeros(2)
    hidden_sequence: int = runtime.stage(TTRLWorkload.train_hidden, hidden)
    reward_sequence: int = runtime.stage(TTRLWorkload.train_rewards, rewards)
    intent_sequence: int = runtime.stage(TTRLWorkload.pending_intent, intent)
    controls: tuple[TTRLClosedLoopIntent, ...] = runtime.drain(TTRLWorkload.control)
    active: TTRLWorkload = runtime.local_workload(workload)
    del hidden_sequence, reward_sequence, intent_sequence, controls, active


def typed_publication(
    runtime: TllmRuntime[ReftWorkload],
    workload: ReftWorkload,
    state: dict[str, torch.Tensor],
) -> int:
    published: ReftParameters = runtime.local_workload(workload).intervention
    del published
    return runtime.publish(ReftWorkload.intervention, state)
