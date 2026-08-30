"""Strict-only public API composition fixture for migrated native workloads."""

from __future__ import annotations

import torch
from typing import cast

from tllm.kit.workload import Workload
from tllm.consumers.dummy.workload import DummyConfig, DummyWorkload
from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload
from tllm.consumers.override.workload import OverRIDEConfig, OverRIDEWorkload
from tllm.consumers.recirculation.workload import (
    RecirculationConfig,
    RecirculationWorkload,
)
from tllm.consumers.reft.workload import ReftConfig, ReftParameters, ReftWorkload
from tllm.consumers.ttrl.rewards import TTRLClosedLoopIntent
from tllm.consumers.ttrl.workload import (
    TTRLConfig,
    TTRLTrainingBatch,
    TTRLTrainingTransaction,
    TTRLWorkload,
    stage_ttrl_training,
)
from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch
from tllm.runtime.retry_control_runtime import RetryControlBatch
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch
from tllm.runtime.request_trainable_state import RequestStateRuntimeService
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.workload import (
    AdamWSpec,
    ConfigField,
    RequestKeyField,
    RequestTrainableState,
    State,
    WorkloadConfig,
    request_key,
    workload,
)

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
    transaction: TTRLTrainingTransaction[TTRLWorkload] = stage_ttrl_training(
        runtime,
        TTRLTrainingBatch(hidden, intent),
    )
    controls: tuple[TTRLClosedLoopIntent, ...] = runtime.drain(TTRLWorkload.control)
    rewards: tuple[RewardOutcomeBatch, ...] = runtime.drain(
        TTRLWorkload.reward_outcomes
    )
    retries: tuple[RetryControlBatch, ...] = runtime.drain(
        TTRLWorkload.retry_controls
    )
    barriers: tuple[UpdateBarrierBatch, ...] = runtime.drain(
        TTRLWorkload.update_barriers
    )
    active: TTRLWorkload = runtime.local_workload(workload)
    del transaction, controls, rewards, retries, barriers, active


def typed_publication(
    runtime: TllmRuntime[ReftWorkload],
    workload: ReftWorkload,
    state: dict[str, torch.Tensor],
) -> int:
    published: ReftParameters = runtime.local_workload(workload).intervention
    del published
    return runtime.publish(ReftWorkload.intervention, state)


_REQUEST_STATE_TYPING_ENABLED = ConfigField.boolean(True)


class _RequestStateTypingConfig(WorkloadConfig):
    enabled: bool = cast(bool, _REQUEST_STATE_TYPING_ENABLED)


class _RequestStateTypingWorkload(Workload[_RequestStateTypingConfig]):
    state: RequestTrainableState[torch.nn.Linear] = State[
        torch.nn.Linear
    ].request_trainable(
        key=request_key(RequestKeyField.REQUEST_ID),
        optimizer=AdamWSpec.fixed(1e-3),
    )

    @state.initialize
    def init_state(self, spec: RuntimeSpecializationSpec) -> torch.nn.Linear:
        return torch.nn.Linear(spec.hidden_size, spec.hidden_size, bias=False)


workload(
    _RequestStateTypingWorkload,
    id="typing.request-trainable",
    config=_RequestStateTypingConfig,
    enabled=_REQUEST_STATE_TYPING_ENABLED,
)


def typed_request_state_service(
    workload_instance: _RequestStateTypingWorkload,
) -> RequestStateRuntimeService[torch.nn.Linear]:
    service: RequestStateRuntimeService[torch.nn.Linear] = workload_instance.state
    return service
