"""TTRL side reward-head workload with typed host staging."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, TypeVar, cast

import torch
from torch.nn import functional as F

from tllm.consumers.ttrl.rewards import (
    TTRLClosedLoopIntent,
    build_ttrl_closed_loop_intent,
)
from tllm.consumers.ttrl.verifier import TTRLVerifier
from tllm.consumers.ttrl.verifier import DEFAULT_TTRL_VERIFIER
from tllm.runtime.generation_trace_runtime import GenerationTraceBatch
from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch
from tllm.runtime.retry_control_runtime import RetryControlBatch
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch
from tllm.kit.workload import Workload
from tllm.workload import (
    AdamSpec,
    Before,
    Between,
    ConfigField,
    ConfigKind,
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
    maybe_optimize,
    read,
    result,
    staged,
    task,
    temporary,
    workload,
)

if TYPE_CHECKING:
    from tllm.runtime.runtime import (
        StagedPortTransactionReceipt,
        TllmRuntime,
    )


_TTRL_ENABLED = ConfigField.boolean(True)
_TTRL_LEARNING_RATE = ConfigField.floating(
    1e-3, NumericConstraint.POSITIVE
)
_RuntimeWorkloadT = TypeVar("_RuntimeWorkloadT")


class TTRLConfig(WorkloadConfig):
    consumer_id: str = config_field(default="ttrl", kind=ConfigKind.STR)
    enabled: bool = cast(bool, _TTRL_ENABLED)
    hidden_size: int = config_field(
        default=0, kind=ConfigKind.INT, constraint=NumericConstraint.NONNEGATIVE
    )
    rank: int = config_field(default=16, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE)
    learning_rate: float = cast(float, _TTRL_LEARNING_RATE)


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
    intent: TTRLClosedLoopIntent


def prepare_ttrl_training_batch(
    trace: GenerationTraceBatch,
    hidden_rows: torch.Tensor,
    *,
    consumer_id: str,
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
    if hidden_rows.ndim != 2 or hidden_rows.shape[0] != len(trace.entries):
        raise ValueError(
            "TTRL hidden rows must align one-to-one with generation trace entries"
        )
    return TTRLTrainingBatch(
        hidden=hidden_rows.detach().clone(),
        intent=intent,
    )


class TTRLWorkload(Workload[TTRLConfig]):
    reward_model: TrainableState[TTRLRewardHead] = State[TTRLRewardHead].replica(
        optimizer=AdamSpec.configured(_TTRL_LEARNING_RATE)
    )
    train_hidden: StagedPort[torch.Tensor] = StagedPort()
    pending_intent: StagedPort[TTRLClosedLoopIntent] = StagedPort()
    control: HostOutput[TTRLClosedLoopIntent] = HostOutput()
    reward_outcomes: HostOutput[RewardOutcomeBatch] = HostOutput()
    retry_controls: HostOutput[RetryControlBatch] = HostOutput()
    update_barriers: HostOutput[UpdateBarrierBatch] = HostOutput()

    @reward_model.initialize
    def init_reward_model(self, spec: RuntimeSpecializationSpec) -> TTRLRewardHead:
        if self.config.hidden_size not in (0, spec.hidden_size):
            raise ValueError(
                "TTRL hidden_size must match the runtime specialization"
            )
        hidden_size = self.config.hidden_size or spec.hidden_size
        return TTRLRewardHead(
            hidden_size, self.config.rank, spec.device, spec.dtype
        )

    @task(
        inputs={
            "hidden": staged(train_hidden),
            "intent": staged(pending_intent),
        },
        outputs={
            "selected_hidden": temporary(),
            "rewards": temporary(),
            "intent": temporary(),
        },
        timing=Between("token.commit", "step.end"),
        implementation=extern("tllm.ttrl:prepare_training_rows@1"),
    )
    def prepare(
        self,
        hidden: torch.Tensor,
        intent: TTRLClosedLoopIntent,
    ) -> tuple[torch.Tensor, torch.Tensor, TTRLClosedLoopIntent]:
        indices = torch.tensor(
            intent.training_indices,
            dtype=torch.long,
            device=hidden.device,
        )
        selected = hidden.index_select(0, indices)
        rewards = torch.tensor(
            intent.training_reward_values,
            dtype=torch.float32,
            device=hidden.device,
        )
        return selected, rewards, intent

    @task(
        inputs={
            "hidden": result("prepare", "selected_hidden"),
            "rewards": result("prepare", "rewards"),
        },
        outputs={
            "loss": maybe_optimize(
                reward_model,
                at=Before("next.prepare"),
            ),
        },
        timing=Between("token.commit", "step.end"),
    )
    def train(
        self,
        hidden: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor | None:
        if hidden.shape[0] == 0:
            return None
        target = rewards.reshape(-1, 1).to(hidden)
        return F.mse_loss(self.reward_model(hidden), target)

    @task(
        inputs={"intent": result("prepare", "intent")},
        outputs={
            "control": host_output(control),
            "reward_outcomes": host_output(reward_outcomes),
            "retry_controls": host_output(retry_controls),
            "update_barriers": host_output(update_barriers),
        },
        effects=(read(reward_model), extern_io("ttrl:control")),
        timing=Before("next.prepare"),
        implementation=extern("tllm.ttrl:publish_control_intent@1"),
    )
    def publish(
        self, intent: TTRLClosedLoopIntent
    ) -> tuple[
        TTRLClosedLoopIntent,
        RewardOutcomeBatch,
        RetryControlBatch,
        UpdateBarrierBatch,
    ]:
        version = (
            f"{self.config.consumer_id}:base"
            if self.state_version == 0
            else f"{self.config.consumer_id}:update:{self.state_version}"
        )
        published = replace(
            intent,
            adapter_version_id=version,
            retry_schedule=tuple(
                replace(entry, adapter_version_id=version)
                for entry in intent.retry_schedule
            ),
        )
        return (
            published,
            published.reward_outcomes,
            published.retry_controls,
            published.update_barriers,
        )


@dataclass(slots=True)
class TTRLTrainingTransaction(Generic[_RuntimeWorkloadT]):
    runtime: TllmRuntime[_RuntimeWorkloadT]
    batch: TTRLTrainingBatch
    staged: StagedPortTransactionReceipt[
        torch.Tensor, TTRLClosedLoopIntent
    ]
    _closed: bool = False

    def commit(self) -> TTRLClosedLoopIntent:
        if self._closed:
            raise RuntimeError("TTRL training transaction is already closed")
        try:
            controls = self.runtime.drain(TTRLWorkload.control)
            rewards = self.runtime.drain(TTRLWorkload.reward_outcomes)
            retries = self.runtime.drain(TTRLWorkload.retry_controls)
            barriers = self.runtime.drain(TTRLWorkload.update_barriers)
        except Exception:
            self._closed = True
            self.runtime.fail_stop_native_workloads(
                "TTRL closed-loop output quorum could not be resolved"
            )
            raise
        if not (
            len(controls)
            == len(rewards)
            == len(retries)
            == len(barriers)
            == 1
        ):
            self._closed = True
            self.runtime.fail_stop_native_workloads(
                "TTRL signed closed-loop outputs diverged"
            )
            raise RuntimeError(
                "TTRL signed execution did not publish one complete closed loop"
            )
        published = controls[0]
        expected = replace(
            self.batch.intent,
            adapter_version_id=published.adapter_version_id,
            retry_schedule=tuple(
                replace(
                    entry,
                    adapter_version_id=published.adapter_version_id,
                )
                for entry in self.batch.intent.retry_schedule
            ),
        )
        if (
            published != expected
            or rewards[0] != published.reward_outcomes
            or retries[0] != published.retry_controls
            or barriers[0] != published.update_barriers
        ):
            self._closed = True
            self.runtime.fail_stop_native_workloads(
                "TTRL host outputs rewrote the signed intent"
            )
            raise RuntimeError("TTRL closed-loop output mismatch")
        try:
            self.runtime.settle_staged_transaction(self.staged)
        except Exception:
            self._closed = True
            self.runtime.fail_stop_native_workloads(
                "TTRL published without consuming its staged transaction"
            )
            raise
        self._closed = True
        return published

    def cancel(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.runtime.cancel_staged_transaction(self.staged)
        except Exception:
            self.runtime.fail_stop_native_workloads(
                "TTRL consumed a transaction that its host cancelled"
            )
            raise


def stage_ttrl_training(
    runtime: TllmRuntime[_RuntimeWorkloadT],
    batch: TTRLTrainingBatch,
) -> TTRLTrainingTransaction[_RuntimeWorkloadT]:
    staged = runtime.stage_pair(
        TTRLWorkload.train_hidden,
        batch.hidden,
        TTRLWorkload.pending_intent,
        batch.intent,
    )
    return TTRLTrainingTransaction(runtime, batch, staged)


workload(TTRLWorkload, id="ttrl", config=TTRLConfig, enabled=_TTRL_ENABLED)


__all__ = [
    "TTRLConfig",
    "TTRLRewardHead",
    "TTRLTrainingBatch",
    "TTRLTrainingTransaction",
    "TTRLWorkload",
    "prepare_ttrl_training_batch",
    "stage_ttrl_training",
]
