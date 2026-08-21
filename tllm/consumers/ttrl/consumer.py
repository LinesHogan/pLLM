#!/usr/bin/env python3
"""TTRL consumer: majority-vote rewards plus a Program-backed reward model.

The verifier/vote/retry/barrier logic is plain Python (``rewards.py``); the
tensor training is a compiled ``TTRLProgram`` driven with caller-supplied
hidden rows and rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from tllm.consumers.base import BaseConsumer
from tllm.consumers._common.bundle_inputs import post_generate_input, provided_bundle
from tllm.consumers.ttrl.config import TTRLConsumerConfig
from tllm.consumers.ttrl.program import TTRLProgram
from tllm.consumers.ttrl.rewards import (
    TTRLClosedLoopIntent,
    TTRLRewardResult,
    TTRLSample,
    build_ttrl_closed_loop_intent,
    build_ttrl_reward_outcome_batch,
    build_ttrl_reward_result,
    majority_vote_rewards,
)
from tllm.consumers.ttrl.verifier import TTRLVerifier, _default_ttrl_verifier
from tllm.ports.base import ConsumerFlow
from tllm.ports.generation_trace import GenerationTrace
from tllm.ports.reward_outcome import RewardOutcome
from tllm.ports.retry_control import RetryControl
from tllm.ports.update_barrier import UpdateBarrier
from tllm.program import ProgramConsumer, StateSpec, compile_program
from tllm.workgraph.frontend import DynamicDim, StateScope


@dataclass(frozen=True)
class TTRLStats:
    update_count: int = 0
    loss_count: int = 0
    reward_count: int = 0
    last_loss: float = 0.0


class TTRLConsumer(BaseConsumer[TTRLConsumerConfig]):
    """Post-generation TTRL-style consumer with local majority rewards."""

    def __init__(
        self,
        config: TTRLConsumerConfig,
        *,
        verifier: TTRLVerifier | None = None,
    ) -> None:
        self.config = config
        self.verifier = verifier or _default_ttrl_verifier()
        self._compiled: ProgramConsumer | None = None
        self._step_id = 0
        self._reward_count = 0
        self._loss_sum_seen = 0.0
        self._last_loss = 0.0

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    @property
    def compiled(self) -> ProgramConsumer | None:
        return self._compiled

    def flows(self) -> Sequence[ConsumerFlow]:
        if not bool(self.config.enabled):
            return ()
        return (
            ConsumerFlow(
                reads=(GenerationTrace.read(scope="completed"),),
                writes=(
                    RewardOutcome.write(scope="completed", mode="append"),
                    RetryControl.write(scope="completed", mode="append"),
                    UpdateBarrier.write(scope="completed", mode="append"),
                ),
                window="out_of_band",
            ),
        )

    # -- rewards (pure Python) ---------------------------------------------

    def rewards_for_samples(self, samples: Sequence[TTRLSample]) -> torch.Tensor:
        rewards = majority_vote_rewards(samples, verifier=self.verifier)
        return torch.tensor(rewards, dtype=torch.float32)

    def reward_result_for_samples(
        self,
        samples: Sequence[TTRLSample],
        *,
        n_votes_per_prompt: int | None = None,
        n_samples_per_prompt: int | None = None,
    ) -> TTRLRewardResult:
        return build_ttrl_reward_result(
            samples,
            n_votes_per_prompt=n_votes_per_prompt,
            n_samples_per_prompt=n_samples_per_prompt,
            verifier=self.verifier,
        )

    def reward_outcome_batch_for_trace(self, trace: object) -> object:
        return build_ttrl_reward_outcome_batch(trace, verifier=self.verifier)

    def closed_loop_intent_for_trace(
        self,
        trace: object,
        *,
        n_votes_per_prompt: int | None = None,
        n_samples_per_prompt: int | None = None,
    ) -> TTRLClosedLoopIntent:
        return build_ttrl_closed_loop_intent(
            trace,
            consumer_id=self.consumer_id,
            n_votes_per_prompt=n_votes_per_prompt,
            n_samples_per_prompt=n_samples_per_prompt,
            verifier=self.verifier,
            adapter_version_id=self.adapter_version_id(),
        )

    def write_reward_outcomes_for_trace(self, runtime: object, trace: object) -> int:
        from tllm.runtime.reward_outcome_runtime import write_reward_outcomes

        return write_reward_outcomes(runtime, build_ttrl_reward_outcome_batch(trace, verifier=self.verifier))

    def consume_bundle(self, bundle, ctx) -> None:
        _ = ctx
        inputs = post_generate_input(bundle)
        intent = self.closed_loop_intent_for_trace(inputs.trace)
        inputs.reward_outcome.append(intent.reward_outcomes)
        if inputs.retry_control is not None:
            inputs.retry_control.append(intent.retry_controls)
        if inputs.update_barrier is not None:
            inputs.update_barrier.append(intent.update_barriers)

    # -- training (compiled program) ----------------------------------------

    def _ensure_compiled(self, hidden: torch.Tensor) -> None:
        if self._compiled is not None:
            return
        if hidden.ndim != 2:
            raise RuntimeError("TTRL update expects 2D hidden rows")
        program = TTRLProgram(
            hidden_size=int(hidden.shape[1]),
            rank=int(self.config.rank),
            lr=float(self.config.learning_rate),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        rows = 2  # export requires example rows strictly above the lower bound
        forward_ex = (
            torch.zeros((rows, int(hidden.shape[1])), device=hidden.device, dtype=hidden.dtype),
            torch.zeros((rows, 1), device=hidden.device, dtype=hidden.dtype),
        )
        predict_ex = (forward_ex[0],)
        row_dims = tuple(
            DynamicDim(value_path=(i,), dimension=0, lower_bound=1, upper_bound=65536)
            for i in range(2)
        )
        self._compiled = compile_program(
            program,
            states={
                "reward_model": StateSpec(scope=StateScope.REPLICA, optimizer=torch.optim.Adam)
            },
            examples={"forward": forward_ex, "predict": predict_ex},
            dynamic_dims={"forward": row_dims, "predict": row_dims[:1]},
        )

    def update_from_rewards(self, hidden: torch.Tensor, rewards: torch.Tensor | Iterable[float]) -> None:
        if not bool(self.config.enabled):
            return
        reward_tensor = (
            rewards
            if isinstance(rewards, torch.Tensor)
            else torch.tensor(list(rewards), dtype=torch.float32)
        )
        rows = int(hidden.shape[0])
        if rows <= 0:
            return
        if int(reward_tensor.reshape(-1).shape[0]) != rows:
            raise RuntimeError("TTRL update requires one reward per hidden row")
        self._ensure_compiled(hidden)
        assert self._compiled is not None
        self._step_id += 1
        bundle = provided_bundle(
            consumer_id=self.consumer_id,
            step_id=self._step_id,
            entries={"hidden": hidden, "rewards": reward_tensor.reshape(-1, 1)},
        )
        self._compiled.consume_flow(0, bundle, None)
        self._reward_count += rows
        loss_sum = float(self._compiled.loss_sum)
        self._last_loss = loss_sum - self._loss_sum_seen
        self._loss_sum_seen = loss_sum

    def predict_scores(self, hidden: torch.Tensor) -> torch.Tensor:
        """Read-only reward predictions against the current bank state."""
        self._ensure_compiled(hidden)
        assert self._compiled is not None
        return self._compiled.run_flow("predict", hidden)["score"]

    def adapter_version_id(self) -> str:
        updates = 0 if self._compiled is None else int(self._compiled.update_count)
        if updates <= 0:
            return f"{self.consumer_id}:base"
        return f"{self.consumer_id}:update:{updates}"

    def read_stats(self) -> TTRLStats:
        compiled = self._compiled
        return TTRLStats(
            update_count=0 if compiled is None else int(compiled.update_count),
            loss_count=0 if compiled is None else int(compiled.loss_count),
            reward_count=int(self._reward_count),
            last_loss=float(self._last_loss),
        )


def run_ttrl_toy_closed_loop(
    consumer: TTRLConsumer,
    trace: object,
    hidden_rows: torch.Tensor,
    *,
    n_votes_per_prompt: int | None = None,
    n_samples_per_prompt: int | None = None,
) -> TTRLClosedLoopIntent:
    from dataclasses import replace

    intent = consumer.closed_loop_intent_for_trace(
        trace,
        n_votes_per_prompt=n_votes_per_prompt,
        n_samples_per_prompt=n_samples_per_prompt,
    )
    rewards = intent.training_rewards()
    if rewards:
        if int(hidden_rows.shape[0]) == len(rewards):
            train_hidden = hidden_rows
        elif intent.training_indices and int(hidden_rows.shape[0]) > max(intent.training_indices):
            train_hidden = hidden_rows[list(intent.training_indices)]
        else:
            raise RuntimeError("hidden rows must align with selected TTRL rewards")
        consumer.update_from_rewards(train_hidden, rewards)
    adapter_version_id = consumer.adapter_version_id()
    retry_schedule = tuple(
        replace(entry, adapter_version_id=adapter_version_id)
        for entry in intent.retry_schedule
    )
    return replace(intent, adapter_version_id=adapter_version_id, retry_schedule=retry_schedule)
