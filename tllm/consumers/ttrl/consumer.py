#!/usr/bin/env python3
"""TTRL-inspired reward grouping and lightweight update consumer."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
from typing import Iterable, Protocol, Sequence

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
from tllm.ports.retry_control import RetryControl
from tllm.ports.update_barrier import UpdateBarrier


_MISSING_BOXED_ANSWER = "missing-boxed-answer"
_TTRL_DEFAULT_VERIFIER_JUDGE = "boxed_normalized_exact"


@dataclass(frozen=True)
class TTRLSample:
    prompt_idx: int
    sample_idx: int
    completion: str


class TTRLVerifier(Protocol):
    def label_for_completion(self, completion: str) -> str:
        """Return a normalized label, or an empty string when unverifiable."""


@dataclass(frozen=True)
class BoxedNormalizedVerifier:
    """Dependency-light verifier for exact matches on normalized boxed answers."""

    judge_name: str = _TTRL_DEFAULT_VERIFIER_JUDGE
    missing_reason: str = _MISSING_BOXED_ANSWER

    def label_for_completion(self, completion: str) -> str:
        return _extract_boxed_answer_label(completion)


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
    verifier_name: str = _TTRL_DEFAULT_VERIFIER_JUDGE


@dataclass(frozen=True)
class TTRLGRPOTrainerBridgeUnavailable:
    """Placeholder for optional external GRPO trainer integrations."""

    reason: str


@dataclass(frozen=True)
class TTRLRetryScheduleEntry:
    request_id: str
    prompt_idx: int
    sample_idx: int
    group_id: str
    adapter_version_id: str


@dataclass(frozen=True)
class TTRLClosedLoopIntent:
    reward_outcomes: object
    retry_controls: object
    update_barriers: object
    training_row_count: int
    selected_reward_count: int = 0
    adapter_version_id: str = ""
    training_indices: tuple[int, ...] = ()
    training_reward_values: tuple[float, ...] = ()
    retry_schedule: tuple[TTRLRetryScheduleEntry, ...] = ()

    def training_rewards(self) -> list[float]:
        return [float(reward) for reward in self.training_reward_values]


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


def _last_boxed_answer_text(text: str) -> str:
    raw = str(text)
    idx = raw.rfind("\\boxed")
    if idx < 0:
        idx = raw.rfind("\\fbox")
    if idx < 0:
        return ""
    left_brace_idx = raw.find("{", idx)
    if left_brace_idx < 0:
        return ""
    depth = 0
    for pos in range(left_brace_idx, len(raw)):
        char = raw[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[left_brace_idx + 1 : pos]
    return ""


def _extract_boxed_answer_label(text: str) -> str:
    answer = _last_boxed_answer_text(str(text))
    if not answer:
        return ""
    return normalize_answer_text(answer)


def _default_ttrl_verifier() -> TTRLVerifier:
    return BoxedNormalizedVerifier()


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
    verifier: TTRLVerifier | None = None,
) -> TTRLRewardResult:
    resolved_verifier = verifier or _default_ttrl_verifier()
    judge_name = str(getattr(resolved_verifier, "judge_name", _TTRL_DEFAULT_VERIFIER_JUDGE))
    missing_reason = str(getattr(resolved_verifier, "missing_reason", _MISSING_BOXED_ANSWER))
    grouped: dict[int, list[tuple[int, TTRLSample, str, str]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        label = str(resolved_verifier.label_for_completion(sample.completion))
        skip_reason = "" if label else missing_reason
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
        total = len(group)
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
        verifier_name=judge_name,
    )


def build_ttrl_reward_outcome_batch(trace: object, *, verifier: TTRLVerifier | None = None) -> object:
    from tllm.runtime.generation_trace_runtime import GenerationTraceBatch
    from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeEntry

    if not isinstance(trace, GenerationTraceBatch):
        raise TypeError("trace must be GenerationTraceBatch")
    resolved_verifier = verifier or _default_ttrl_verifier()
    samples = [
        TTRLSample(
            prompt_idx=int(entry.prompt_idx),
            sample_idx=int(entry.sample_idx),
            completion=str(entry.text),
        )
        for entry in trace.entries
    ]
    result = build_ttrl_reward_result(samples, verifier=resolved_verifier)
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
                judge=f"ttrl_majority:{result.verifier_name}",
                label="" if skipped else str(sample_result.label),
                skip_reason=str(sample_result.skip_reason),
                is_correct=None if skipped else bool(sample_result.reward == 1.0),
            )
        )
    return RewardOutcomeBatch(entries=tuple(outcome_entries))


def _ttrl_barrier_id(group_id: str, prompt_idx: int) -> str:
    return f"ttrl:{group_id}" if group_id else f"ttrl:prompt:{int(prompt_idx)}"


def build_ttrl_closed_loop_intent(
    trace: object,
    *,
    consumer_id: str = "ttrl",
    n_votes_per_prompt: int | None = None,
    n_samples_per_prompt: int | None = None,
    verifier: TTRLVerifier | None = None,
    adapter_version_id: str = "",
) -> TTRLClosedLoopIntent:
    from tllm.runtime.generation_trace_runtime import GenerationTraceBatch
    from tllm.runtime.retry_control_runtime import RetryControlBatch, RetryControlEntry
    from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch, UpdateBarrierEntry

    if not isinstance(trace, GenerationTraceBatch):
        raise TypeError("trace must be GenerationTraceBatch")

    resolved_verifier = verifier or _default_ttrl_verifier()
    reward_outcomes = build_ttrl_reward_outcome_batch(trace, verifier=resolved_verifier)
    samples = [
        TTRLSample(
            prompt_idx=int(entry.prompt_idx),
            sample_idx=int(entry.sample_idx),
            completion=str(entry.text),
        )
        for entry in trace.entries
    ]
    reward_result = build_ttrl_reward_result(
        samples,
        n_votes_per_prompt=n_votes_per_prompt,
        n_samples_per_prompt=n_samples_per_prompt,
        verifier=resolved_verifier,
    )
    selected_train_indices = {
        int(idx)
        for idx in reward_result.selected_indices
        if reward_outcomes.entries[int(idx)].reward is not None
    }
    retry_entries: list[RetryControlEntry] = []
    grouped: dict[tuple[str, int], list[object]] = defaultdict(list)
    grouped_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    training_reward_values: list[float] = []
    training_indices: list[int] = []
    for entry_idx, (trace_entry, outcome) in enumerate(zip(trace.entries, reward_outcomes.entries, strict=True)):
        grouped[(str(trace_entry.group_id), int(trace_entry.prompt_idx))].append(outcome)
        grouped_indices[(str(trace_entry.group_id), int(trace_entry.prompt_idx))].append(int(entry_idx))
        if outcome.reward is None:
            action = "retry"
            reason = outcome.skip_reason or "missing-reward"
            retry_after_update = False
        elif bool(outcome.is_correct):
            action = "accept"
            reason = "majority-answer"
            retry_after_update = False
            if int(entry_idx) in selected_train_indices:
                training_reward_values.append(float(outcome.reward))
                training_indices.append(int(entry_idx))
        else:
            action = "retry"
            reason = "non-majority-answer"
            retry_after_update = True
            if int(entry_idx) in selected_train_indices:
                training_reward_values.append(float(outcome.reward))
                training_indices.append(int(entry_idx))
        retry_entries.append(
            RetryControlEntry(
                request_id=str(outcome.request_id),
                prompt_idx=int(outcome.prompt_idx),
                sample_idx=int(outcome.sample_idx),
                output_index=int(outcome.output_index),
                group_id=str(outcome.group_id),
                action=action,
                reason=reason,
                retry_after_update=retry_after_update,
            )
        )

    barrier_entries: list[UpdateBarrierEntry] = []
    for (group_id, prompt_idx), outcomes in grouped.items():
        ready_count = sum(1 for idx in grouped_indices[(group_id, prompt_idx)] if idx in selected_train_indices)
        barrier_entries.append(
            UpdateBarrierEntry(
                barrier_id=_ttrl_barrier_id(group_id, prompt_idx),
                group_id=group_id,
                consumer_id=str(consumer_id),
                state="ready" if ready_count else "blocked",
                request_ids=tuple(str(outcome.request_id) for outcome in outcomes),
                ready_count=int(ready_count),
                reason="majority-reward-ready" if ready_count else "no-trainable-reward",
            )
        )

    retry_schedule = tuple(
        TTRLRetryScheduleEntry(
            request_id=str(entry.request_id),
            prompt_idx=int(entry.prompt_idx),
            sample_idx=int(entry.sample_idx),
            group_id=str(entry.group_id),
            adapter_version_id=str(adapter_version_id),
        )
        for entry in retry_entries
        if bool(entry.retry_after_update)
    )

    return TTRLClosedLoopIntent(
        reward_outcomes=reward_outcomes,
        retry_controls=RetryControlBatch(entries=tuple(retry_entries)),
        update_barriers=UpdateBarrierBatch(entries=tuple(barrier_entries)),
        training_row_count=len(training_reward_values),
        selected_reward_count=len(training_reward_values),
        adapter_version_id=str(adapter_version_id),
        training_indices=tuple(training_indices),
        training_reward_values=tuple(training_reward_values),
        retry_schedule=retry_schedule,
    )


def majority_vote_rewards(samples: Sequence[TTRLSample], *, verifier: TTRLVerifier | None = None) -> list[float]:
    return build_ttrl_reward_result(samples, verifier=verifier).rewards


def _tuple_float_values(values: object, *, field_name: str) -> tuple[float, ...]:
    if values is None:
        return ()
    if hasattr(values, "detach") and callable(getattr(values, "detach")):
        values = values.detach().cpu().reshape(-1).tolist()
    try:
        return tuple(float(value) for value in values)  # type: ignore[union-attr]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of real values") from exc


def _logprobs_for_index(
    trace_entry: object,
    index: int,
    mapping: Mapping[int, Sequence[float]] | None,
    *,
    attr_name: str,
    field_name: str,
) -> tuple[float, ...] | None:
    values: object | None = None
    if mapping is not None and int(index) in mapping:
        values = mapping[int(index)]
    elif hasattr(trace_entry, attr_name):
        values = getattr(trace_entry, attr_name)
    if values is None:
        return None
    normalized = _tuple_float_values(values, field_name=field_name)
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty when provided")
    return normalized


def build_ttrl_grpo_training_batch(
    trace: object,
    *,
    n_votes_per_prompt: int | None = None,
    n_samples_per_prompt: int | None = None,
    verifier: TTRLVerifier | None = None,
    current_logprobs_by_index: Mapping[int, Sequence[float]] | None = None,
    reference_logprobs_by_index: Mapping[int, Sequence[float]] | None = None,
) -> object:
    """Build dependency-light GRPO examples from a completed generation trace."""

    from tllm.contracts.test_time_training import TestTimeTrainingBatch, TestTimeTrainingExample
    from tllm.runtime.generation_trace_runtime import GenerationTraceBatch

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
    reward_result = build_ttrl_reward_result(
        samples,
        n_votes_per_prompt=n_votes_per_prompt,
        n_samples_per_prompt=n_samples_per_prompt,
        verifier=verifier,
    )

    examples: list[TestTimeTrainingExample] = []
    selected_training_indices: list[int] = []
    for training_order, sample_index in enumerate(reward_result.selected_indices):
        trace_entry = trace.entries[int(sample_index)]
        token_ids = tuple(int(token_id) for token_id in trace_entry.token_ids)
        if not token_ids:
            raise ValueError("GRPO training examples require non-empty token_ids")
        sample_result = reward_result.sample_results[int(training_order)]
        current_logprobs = _logprobs_for_index(
            trace_entry,
            int(sample_index),
            current_logprobs_by_index,
            attr_name="logprobs",
            field_name="current_logprobs",
        )
        reference_logprobs = _logprobs_for_index(
            trace_entry,
            int(sample_index),
            reference_logprobs_by_index,
            attr_name="reference_logprobs",
            field_name="reference_logprobs",
        )
        if current_logprobs is not None and len(current_logprobs) != len(token_ids):
            raise ValueError("current_logprobs length must match token_ids")
        if reference_logprobs is not None and len(reference_logprobs) != len(token_ids):
            raise ValueError("reference_logprobs length must match token_ids")
        metadata: dict[str, object] = {
            "group_id": str(trace_entry.group_id),
            "training_index": int(sample_index),
            "prompt_idx": int(trace_entry.prompt_idx),
            "sample_idx": int(trace_entry.sample_idx),
            "output_index": int(trace_entry.output_index),
            "majority_label": str(sample_result.majority_label),
            "sample_label": str(sample_result.label),
            "majority_ratio": float(sample_result.majority_ratio),
            "finish_reason": str(trace_entry.finish_reason),
        }
        if reference_logprobs is not None:
            metadata["reference_logprobs"] = reference_logprobs
        examples.append(
            TestTimeTrainingExample(
                request_id=str(trace_entry.request_id),
                prompt_id=str(trace_entry.prompt_idx),
                sample_id=str(trace_entry.sample_idx),
                output_id=str(trace_entry.output_index),
                objective_kind="grpo",
                token_ids=token_ids,
                rewards=(float(sample_result.reward),),
                logprobs=current_logprobs,
                metadata=metadata,
            )
        )
        selected_training_indices.append(int(sample_index))

    return TestTimeTrainingBatch(
        examples=tuple(examples),
        metadata={
            "algorithm_id": "ttrl_grpo",
            "selected_training_indices": tuple(selected_training_indices),
            "verifier_name": str(reward_result.verifier_name),
        },
    )


def ttrl_grpo_group_normalized_advantages(
    rewards: torch.Tensor,
    group_ids: Sequence[object],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize scalar rewards within each group for toy GRPO tests."""

    reward_values = rewards.reshape(-1)
    if not torch.is_floating_point(reward_values):
        reward_values = reward_values.to(dtype=torch.float32)
    if len(group_ids) != int(reward_values.shape[0]):
        raise ValueError("group_ids length must match rewards")
    advantages = torch.zeros_like(reward_values)
    grouped_indices: dict[object, list[int]] = defaultdict(list)
    for idx, group_id in enumerate(group_ids):
        grouped_indices[group_id].append(int(idx))
    for indices in grouped_indices.values():
        index_tensor = torch.tensor(indices, device=reward_values.device)
        group_rewards = reward_values.index_select(0, index_tensor)
        centered = group_rewards - group_rewards.mean()
        std = group_rewards.std(unbiased=False)
        if float(std.detach().cpu()) <= float(eps):
            group_advantages = torch.zeros_like(group_rewards)
        else:
            group_advantages = centered / (std + float(eps))
        advantages.index_copy_(0, index_tensor, group_advantages)
    return advantages.reshape_as(rewards)


def ttrl_grpo_clipped_ratio_loss(
    *,
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a PPO/GRPO-style clipped ratio loss for small logprob tensors."""

    if current_logprobs.shape != old_logprobs.shape:
        raise ValueError("current_logprobs and old_logprobs must have the same shape")
    if current_logprobs.ndim == 0:
        raise ValueError("logprob tensors must have at least one dimension")
    token_delta = current_logprobs - old_logprobs
    if mask is not None:
        if mask.shape != current_logprobs.shape:
            raise ValueError("mask shape must match logprob tensors")
        token_delta = token_delta * mask.to(device=current_logprobs.device, dtype=current_logprobs.dtype)
    sample_log_ratio = token_delta.reshape(int(token_delta.shape[0]), -1).sum(dim=1)
    flat_advantages = advantages.reshape(-1).to(device=current_logprobs.device, dtype=current_logprobs.dtype)
    if int(flat_advantages.shape[0]) != int(sample_log_ratio.shape[0]):
        raise ValueError("advantages length must match logprob batch size")
    ratio = torch.exp(sample_log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    surrogate = torch.minimum(ratio * flat_advantages, clipped_ratio * flat_advantages)
    return -surrogate.mean()


def create_optional_verl_grpo_bridge(*args: object, **kwargs: object) -> TTRLGRPOTrainerBridgeUnavailable:
    """Placeholder factory for optional verl-backed GRPO training.

    Core TTRL code intentionally does not import verl. A real integration should
    live in an optional package and supply the shared TestTimeTrainerBridge
    protocol.
    """

    _ = args, kwargs
    raise RuntimeError("optional verl integration is unavailable; install and load an external trainer bridge")


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

    def adapter_version_id(self, *, consumer_id: str = "ttrl") -> str:
        if int(self._update_count) <= 0:
            return f"{consumer_id}:base"
        return f"{consumer_id}:update:{int(self._update_count)}"


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
