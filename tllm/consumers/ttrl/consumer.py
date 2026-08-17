#!/usr/bin/env python3
"""TTRL-inspired reward grouping and lightweight update consumer."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

import torch

from tllm.consumers.base import BaseConsumer
from tllm.consumers.ttrl.config import TTRLConsumerConfig
from tllm.consumers.ttrl.engine import (
    TTRLClosedLoopIntent,
    TTRLEngine,
    TTRLRewardResult,
    TTRLSample,
    TTRLStats,
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


class TTRLConsumer(BaseConsumer):
    """Post-generation TTRL-style consumer with local majority rewards."""

    def __init__(
        self,
        config: TTRLConsumerConfig,
        *,
        engine: TTRLEngine | None = None,
        verifier: TTRLVerifier | None = None,
    ) -> None:
        self.config = config
        self.engine = engine or TTRLEngine(config)
        self.verifier = verifier or _default_ttrl_verifier()

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

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
            adapter_version_id=self.engine.adapter_version_id(consumer_id=self.consumer_id),
        )

    def write_reward_outcomes_for_trace(self, runtime: object, trace: object) -> int:
        from tllm.runtime.reward_outcome_runtime import write_reward_outcomes

        return write_reward_outcomes(runtime, build_ttrl_reward_outcome_batch(trace, verifier=self.verifier))

    def consume_bundle(self, bundle, ctx) -> None:
        _ = ctx
        from tllm.runtime.generation_trace_runtime import GenerationTraceBatch

        trace = bundle.entries.get("generation_trace")
        if not isinstance(trace, GenerationTraceBatch):
            raise TypeError("generation_trace entry must be GenerationTraceBatch")
        sink = bundle.entries.get("reward_outcome")
        append = getattr(sink, "append", None)
        if not callable(append):
            raise TypeError("reward_outcome entry must provide append")
        intent = self.closed_loop_intent_for_trace(trace)
        append(intent.reward_outcomes)
        retry_sink = bundle.entries.get("retry_control")
        retry_append = getattr(retry_sink, "append", None)
        if retry_sink is not None and not callable(retry_append):
            raise TypeError("retry_control entry must provide append")
        if callable(retry_append):
            retry_append(intent.retry_controls)
        barrier_sink = bundle.entries.get("update_barrier")
        barrier_append = getattr(barrier_sink, "append", None)
        if barrier_sink is not None and not callable(barrier_append):
            raise TypeError("update_barrier entry must provide append")
        if callable(barrier_append):
            barrier_append(intent.update_barriers)

    def update_from_rewards(self, hidden: torch.Tensor, rewards: torch.Tensor | Iterable[float]) -> None:
        reward_tensor = rewards if isinstance(rewards, torch.Tensor) else torch.tensor(list(rewards), dtype=torch.float32)
        self.engine.update(hidden, reward_tensor)

    def submit_grpo_training_batch(
        self,
        batch: object,
        *,
        trainer_bridge: object,
        lora_service: object | None = None,
        adapter_key: str = "ttrl-grpo",
        target_modules: tuple[str, ...] = (),
        scope: str = "request",
        request_ids: tuple[str, ...] | None = None,
    ) -> object:
        from tllm.contracts.test_time_training import TestTimeTrainerRequest, TestTimeTrainingBatch
        from tllm.runtime.test_time.lora import TestTimeLoRAService
        from tllm.runtime.test_time.trainer import stage_trainer_result_updates

        if not isinstance(batch, TestTimeTrainingBatch):
            raise TypeError("batch must be TestTimeTrainingBatch")
        if not target_modules:
            raise ValueError("target_modules must be non-empty")
        train = getattr(trainer_bridge, "train", None)
        if not callable(train):
            raise TypeError("trainer_bridge must provide train")
        resolved_request_ids = tuple(batch.request_ids if request_ids is None else request_ids)
        request = TestTimeTrainerRequest(
            algorithm_id="ttrl_grpo",
            adapter_key=str(adapter_key),
            target_modules=tuple(str(module_name) for module_name in target_modules),
            rank=int(self.config.rank),
            scope=scope,  # type: ignore[arg-type]
            request_ids=resolved_request_ids,
            examples=batch,
            metadata={"consumer_id": self.consumer_id},
        )
        result = train(request)
        if lora_service is not None:
            if not isinstance(lora_service, TestTimeLoRAService):
                raise TypeError("lora_service must be TestTimeLoRAService")
            stage_trainer_result_updates(request, result, lora_service)
        return result

    def read_stats(self) -> TTRLStats:
        return self.engine.read_stats()


def run_ttrl_toy_closed_loop(
    consumer: TTRLConsumer,
    trace: object,
    hidden_rows: torch.Tensor,
    *,
    n_votes_per_prompt: int | None = None,
    n_samples_per_prompt: int | None = None,
) -> TTRLClosedLoopIntent:
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
    adapter_version_id = consumer.engine.adapter_version_id(consumer_id=consumer.consumer_id)
    retry_schedule = tuple(
        replace(entry, adapter_version_id=adapter_version_id)
        for entry in intent.retry_schedule
    )
    return replace(intent, adapter_version_id=adapter_version_id, retry_schedule=retry_schedule)
