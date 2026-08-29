"""TTRL side reward-head workload with typed host staging."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

import torch
from torch.nn import functional as F

from tllm.consumers.ttrl.rewards import (
    TTRLClosedLoopIntent,
    build_ttrl_closed_loop_intent,
)
from tllm.consumers.ttrl.verifier import TTRLVerifier
from tllm.consumers.ttrl.verifier import DEFAULT_TTRL_VERIFIER
from tllm.runtime.generation_trace_runtime import GenerationTraceBatch
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.kit.workload import Workload
from tllm.workload import (
    AdamSpec,
    Before,
    Between,
    ConfigField,
    HostOutput,
    NumericConstraint,
    State,
    TrainableState,
    StagedPort,
    WorkloadConfig,
    config_field,
    extern,
    extern_io,
    host_output,
    optimize,
    read,
    staged,
    task,
    workload,
)


_TTRL_ENABLED = ConfigField(True)


class TTRLConfig(WorkloadConfig):
    consumer_id: str = config_field(default="ttrl")
    enabled: bool = cast(bool, _TTRL_ENABLED)
    hidden_size: int = config_field(
        default=0, constraint=NumericConstraint.NONNEGATIVE
    )
    rank: int = config_field(default=16, constraint=NumericConstraint.POSITIVE)
    learning_rate: float = config_field(
        default=1e-3, constraint=NumericConstraint.POSITIVE
    )


class TTRLRewardHead(torch.nn.Module):
    def __init__(
        self, hidden_size: int, rank: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.down = torch.nn.Parameter(
            torch.empty((rank, hidden_size), device=device, dtype=dtype)
        )
        torch.nn.init.normal_(
            self.down, mean=0.0, std=1.0 / float(hidden_size) ** 0.5
        )
        self.up = torch.nn.Parameter(
            torch.zeros((1, rank), device=device, dtype=dtype)
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(hidden, self.down), self.up)


@dataclass(frozen=True, slots=True)
class TTRLTrainingBatch:
    hidden: torch.Tensor
    rewards: torch.Tensor
    intent: TTRLClosedLoopIntent


def prepare_ttrl_training_batch(
    trace: GenerationTraceBatch,
    hidden_rows: torch.Tensor,
    *,
    consumer_id: str = "ttrl",
    n_votes_per_prompt: int = 0,
    n_samples_per_prompt: int = 0,
    verifier: TTRLVerifier = DEFAULT_TTRL_VERIFIER,
) -> TTRLTrainingBatch:
    """Convert one external generation trace at the runtime boundary."""

    intent = build_ttrl_closed_loop_intent(
        trace,
        consumer_id=consumer_id,
        n_votes_per_prompt=n_votes_per_prompt,
        n_samples_per_prompt=n_samples_per_prompt,
        verifier=verifier,
    )
    rewards = intent.training_rewards()
    selected = (
        hidden_rows
        if hidden_rows.shape[0] == len(rewards)
        else hidden_rows[list(intent.training_indices)]
    )
    return TTRLTrainingBatch(
        hidden=selected.detach().clone(),
        rewards=torch.tensor(rewards, dtype=torch.float32, device=selected.device),
        intent=intent,
    )


class TTRLWorkload(Workload[TTRLConfig]):
    reward_model: State[TTRLRewardHead] = State[TTRLRewardHead].replica(
        optimizer=AdamSpec(lr=TTRLConfig.learning_rate)
    )
    train_hidden: StagedPort[torch.Tensor] = StagedPort()
    train_rewards: StagedPort[torch.Tensor] = StagedPort()
    pending_intent: StagedPort[TTRLClosedLoopIntent] = StagedPort()
    control: HostOutput[TTRLClosedLoopIntent] = HostOutput()

    @reward_model.initialize
    def init_reward_model(self, spec: RuntimeSpecializationSpec) -> TTRLRewardHead:
        hidden_size = self.config.hidden_size or spec.hidden_size
        return TTRLRewardHead(
            hidden_size, self.config.rank, spec.device, spec.dtype
        )

    @task(
        inputs={
            "hidden": staged(train_hidden),
            "rewards": staged(train_rewards),
        },
        outputs={
            "loss": optimize(
                cast(TrainableState[object], reward_model),
                at=Before("next.prepare"),
            )
        },
        timing=Between("token.commit", "step.end"),
    )
    def train(self, hidden: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        target = rewards.reshape(-1, 1).to(hidden)
        return F.mse_loss(self.reward_model(hidden), target)

    @task(
        inputs={"intent": staged(pending_intent)},
        outputs={"control": host_output(control)},
        effects=(read(reward_model), extern_io("ttrl:control")),
        timing=Before("next.prepare"),
        implementation=extern("tllm.ttrl:publish_control_intent@1"),
    )
    def publish(self, intent: TTRLClosedLoopIntent) -> TTRLClosedLoopIntent:
        version = f"{self.config.consumer_id}:update:{self.state_version}"
        return replace(
            intent,
            adapter_version_id=version,
            retry_schedule=tuple(
                replace(entry, adapter_version_id=version)
                for entry in intent.retry_schedule
            ),
        )


workload(TTRLWorkload, id="ttrl", config=TTRLConfig, enabled=_TTRL_ENABLED)


__all__ = [
    "TTRLConfig",
    "TTRLRewardHead",
    "TTRLTrainingBatch",
    "TTRLWorkload",
    "prepare_ttrl_training_batch",
]
