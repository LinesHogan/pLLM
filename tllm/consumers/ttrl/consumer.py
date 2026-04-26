#!/usr/bin/env python3
"""TTRL-inspired reward grouping and lightweight update consumer."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tllm.consumers.base import BaseConsumer
from tllm.consumers._common.alignment import (
    LowRankAdapter,
    normalize_answer_text,
    select_top_k_per_prompt,
)
from tllm.ports.base import ConsumerFlow
from tllm.ports.generation_trace import GenerationTrace
from tllm.ports.reward_outcome import RewardOutcome


_BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_MISSING_BOXED_ANSWER = "missing-boxed-answer"
_TTRL_BOXED_MAJORITY_JUDGE = "ttrl_boxed_majority"


@dataclass(frozen=True)
class TTRLSample:
    prompt_idx: int
    sample_idx: int
    completion: str


@dataclass(frozen=True)
class TTRLMajorityResult:
    prompt_idx: int
    label: str
    ratio: float
    count: int
    total: int


@dataclass(frozen=True)
class TTRLSampleReward:
    prompt_idx: int
    sample_idx: int
    completion: str
    label: str
    majority_label: str
    majority_ratio: float
    reward: float
    skip_reason: str = ""


@dataclass(frozen=True)
class TTRLRewardResult:
    rewards: list[float]
    majority_by_prompt: dict[int, TTRLMajorityResult]
    sample_results: list[TTRLSampleReward]
    selected_indices: list[int]


@dataclass
class TTRLConsumerConfig:
    consumer_id: str = "ttrl"
    enabled: bool = True
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class TTRLStats:
    update_count: int = 0
    loss_count: int = 0
    reward_count: int = 0
    last_loss: float = 0.0


def _extract_boxed_answer_label(text: str) -> str:
    matches = _BOXED_ANSWER_PATTERN.findall(str(text))
    if not matches:
        return ""
    return normalize_answer_text(matches[-1])


def select_ttrl_training_indices(
    *,
    num_prompts: int,
    n_votes_per_prompt: int,
    n_samples_per_prompt: int,
) -> list[int]:
    return select_top_k_per_prompt(
        num_prompts=int(num_prompts),
        votes_per_prompt=int(n_votes_per_prompt),
        samples_per_prompt=int(n_samples_per_prompt),
    )


def build_ttrl_reward_result(
    samples: Sequence[TTRLSample],
    *,
    n_votes_per_prompt: int | None = None,
    n_samples_per_prompt: int | None = None,
) -> TTRLRewardResult:
    grouped: dict[int, list[tuple[int, TTRLSample, str, str]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        label = _extract_boxed_answer_label(sample.completion)
        skip_reason = "" if label else _MISSING_BOXED_ANSWER
        grouped[int(sample.prompt_idx)].append((idx, sample, label, skip_reason))

    majority_by_prompt: dict[int, TTRLMajorityResult] = {}
    result_by_input_idx: dict[int, TTRLSampleReward] = {}
    for prompt_idx, group in grouped.items():
        counts = Counter(label for _, _, label, skip_reason in group if not skip_reason)
        best_count = max(counts.values()) if counts else 0
        majority_label = ""
        for _, _, label, skip_reason in group:
            if not skip_reason and counts[label] == best_count:
                majority_label = label
                break
        total = sum(1 for _, _, _, skip_reason in group if not skip_reason)
        majority_ratio = (float(best_count) / float(total)) if total else 0.0
        majority_by_prompt[int(prompt_idx)] = TTRLMajorityResult(
            prompt_idx=int(prompt_idx),
            label=majority_label,
            ratio=majority_ratio,
            count=int(best_count),
            total=int(total),
        )
        for idx, sample, label, skip_reason in group:
            reward = 1.0 if label and label == majority_label else 0.0
            sample_result = TTRLSampleReward(
                prompt_idx=int(sample.prompt_idx),
                sample_idx=int(sample.sample_idx),
                completion=str(sample.completion),
                label=label,
                majority_label=majority_label,
                majority_ratio=majority_ratio,
                reward=reward,
                skip_reason=skip_reason,
            )
            result_by_input_idx[int(idx)] = sample_result

    if n_votes_per_prompt is None and n_samples_per_prompt is None:
        selected_indices = list(range(len(samples)))
    elif n_votes_per_prompt is not None and n_samples_per_prompt is not None:
        if int(n_votes_per_prompt) <= 0:
            raise ValueError("n_votes_per_prompt must be positive")
        if len(samples) % int(n_votes_per_prompt) != 0:
            raise ValueError("sample count must be divisible by n_votes_per_prompt")
        selected_indices = select_ttrl_training_indices(
            num_prompts=len(samples) // int(n_votes_per_prompt),
            n_votes_per_prompt=int(n_votes_per_prompt),
            n_samples_per_prompt=int(n_samples_per_prompt),
        )
    else:
        raise ValueError("n_votes_per_prompt and n_samples_per_prompt must be provided together")

    sample_results = [result_by_input_idx[idx] for idx in selected_indices]
    rewards = [sample_result.reward for sample_result in sample_results]
    return TTRLRewardResult(
        rewards=rewards,
        majority_by_prompt=majority_by_prompt,
        sample_results=sample_results,
        selected_indices=selected_indices,
    )


def build_ttrl_reward_outcome_batch(trace: object) -> object:
    from tllm.runtime.generation_trace_runtime import GenerationTraceBatch
    from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeEntry

    if not isinstance(trace, GenerationTraceBatch):
        raise TypeError("trace must be GenerationTraceBatch")
    samples = [
        TTRLSample(
            prompt_idx=int(entry.prompt_idx),
            sample_idx=int(entry.sample_idx),
            completion=str(entry.text),
        )
        for entry in trace.entries
    ]
    result = build_ttrl_reward_result(samples)
    outcome_entries = []
    for trace_entry, sample_result in zip(trace.entries, result.sample_results, strict=True):
        skipped = bool(sample_result.skip_reason)
        outcome_entries.append(
            RewardOutcomeEntry(
                request_id=str(trace_entry.request_id),
                prompt_idx=int(trace_entry.prompt_idx),
                sample_idx=int(trace_entry.sample_idx),
                output_index=int(trace_entry.output_index),
                group_id=str(trace_entry.group_id),
                reward=None if skipped else float(sample_result.reward),
                judge=_TTRL_BOXED_MAJORITY_JUDGE,
                label="" if skipped else str(sample_result.label),
                skip_reason=str(sample_result.skip_reason),
                is_correct=None if skipped else bool(sample_result.reward == 1.0),
            )
        )
    return RewardOutcomeBatch(entries=tuple(outcome_entries))


def majority_vote_rewards(samples: Sequence[TTRLSample]) -> list[float]:
    return build_ttrl_reward_result(samples).rewards


class TTRLEngine(nn.Module):
    def __init__(self, config: TTRLConsumerConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter: LowRankAdapter | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self._update_count = 0
        self._loss_count = 0
        self._reward_count = 0
        self._last_loss = 0.0
        if int(config.hidden_size) > 0:
            self._build_adapter(int(config.hidden_size))

    @staticmethod
    def _detached_train_input(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach()
        if torch.is_inference(out):
            out = out.clone()
        return out

    def _build_adapter(self, hidden_size: int) -> None:
        self.config.hidden_size = int(hidden_size)
        self.adapter = LowRankAdapter(in_dim=int(hidden_size), out_dim=1, rank=int(self.config.rank))
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=float(self.config.learning_rate))

    def ensure_shape(self, hidden: torch.Tensor) -> None:
        if hidden.ndim != 2:
            raise RuntimeError("TTRL update expects 2D hidden rows")
        hidden_size = int(hidden.shape[1])
        if self.adapter is None:
            self._build_adapter(hidden_size)
        elif int(self.config.hidden_size) != hidden_size:
            raise RuntimeError("TTRL hidden size changed after adapter initialization")
        assert self.adapter is not None
        first_param = next(self.adapter.parameters(), None)
        if first_param is not None and (first_param.device != hidden.device or first_param.dtype != hidden.dtype):
            self.to(device=hidden.device, dtype=hidden.dtype)

    def update(self, hidden: torch.Tensor, rewards: torch.Tensor) -> None:
        if not bool(self.config.enabled):
            return
        if int(hidden.shape[0]) != int(rewards.reshape(-1).shape[0]):
            raise RuntimeError("TTRL update requires one reward per hidden row")
        if int(hidden.shape[0]) <= 0:
            return
        self.ensure_shape(hidden)
        assert self.adapter is not None
        assert self.optimizer is not None
        hidden_train = self._detached_train_input(hidden)
        rewards_train = self._detached_train_input(rewards.reshape(-1, 1).to(device=hidden.device, dtype=hidden.dtype))
        pred = self.adapter(hidden_train)
        loss = F.mse_loss(pred, rewards_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_count += 1
        self._loss_count += 1
        self._reward_count += int(rewards_train.shape[0])
        self._last_loss = float(loss.detach().cpu())

    def read_stats(self) -> TTRLStats:
        return TTRLStats(
            update_count=int(self._update_count),
            loss_count=int(self._loss_count),
            reward_count=int(self._reward_count),
            last_loss=float(self._last_loss),
        )


class TTRLConsumer(BaseConsumer):
    """Post-generation TTRL-style consumer with local majority rewards."""

    def __init__(self, config: TTRLConsumerConfig, *, engine: TTRLEngine | None = None) -> None:
        self.config = config
        self.engine = engine or TTRLEngine(config)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        if not bool(self.config.enabled):
            return ()
        return (
            ConsumerFlow(
                reads=(GenerationTrace.read(scope="completed"),),
                writes=(RewardOutcome.write(scope="completed", mode="append"),),
                window="out_of_band",
            ),
        )

    def rewards_for_samples(self, samples: Sequence[TTRLSample]) -> torch.Tensor:
        rewards = majority_vote_rewards(samples)
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
        )

    def reward_outcome_batch_for_trace(self, trace: object) -> object:
        return build_ttrl_reward_outcome_batch(trace)

    def write_reward_outcomes_for_trace(self, runtime: object, trace: object) -> int:
        from tllm.runtime.reward_outcome_runtime import write_reward_outcomes

        return write_reward_outcomes(runtime, build_ttrl_reward_outcome_batch(trace))

    def update_from_rewards(self, hidden: torch.Tensor, rewards: torch.Tensor | Iterable[float]) -> None:
        reward_tensor = rewards if isinstance(rewards, torch.Tensor) else torch.tensor(list(rewards), dtype=torch.float32)
        self.engine.update(hidden, reward_tensor)

    def read_stats(self) -> TTRLStats:
        return self.engine.read_stats()
