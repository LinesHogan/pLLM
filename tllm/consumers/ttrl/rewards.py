#!/usr/bin/env python3
"""Pure TTRL majority rewards and typed closed-loop intent construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from tllm.consumers.ttrl.verifier import (
    TTRL_DEFAULT_VERIFIER_JUDGE,
    TTRLVerifier,
    DEFAULT_TTRL_VERIFIER,
)
from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch
from tllm.runtime.retry_control_runtime import RetryControlBatch
from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch

if TYPE_CHECKING:
    from tllm.runtime.generation_trace_runtime import GenerationTraceBatch
    from tllm.runtime.reward_outcome_runtime import RewardOutcomeEntry


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
    verifier_name: str = TTRL_DEFAULT_VERIFIER_JUDGE


@dataclass(frozen=True)
class TTRLRetryScheduleEntry:
    request_id: str
    prompt_idx: int
    sample_idx: int
    output_index: int
    group_id: str
    adapter_version_id: str


@dataclass(frozen=True)
class TTRLClosedLoopIntent:
    reward_outcomes: RewardOutcomeBatch
    retry_controls: RetryControlBatch
    update_barriers: UpdateBarrierBatch
    training_row_count: int
    selected_reward_count: int = 0
    adapter_version_id: str = ""
    training_indices: tuple[int, ...] = ()
    training_reward_values: tuple[float, ...] = ()
    retry_schedule: tuple[TTRLRetryScheduleEntry, ...] = ()

def select_ttrl_training_indices(
    *,
    samples: Sequence[TTRLSample],
    n_votes_per_prompt: int,
    n_samples_per_prompt: int,
) -> list[int]:
    if n_votes_per_prompt <= 0 or n_samples_per_prompt <= 0:
        raise ValueError("TTRL vote and sample counts must be positive")
    if n_samples_per_prompt > n_votes_per_prompt:
        raise ValueError("TTRL samples per prompt cannot exceed vote count")
    grouped: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        grouped.setdefault(sample.prompt_idx, []).append(index)
    if any(
        len(indices) != n_votes_per_prompt
        for indices in grouped.values()
    ):
        raise ValueError(
            "each TTRL prompt must contain the declared vote count"
        )
    return [
        index
        for indices in grouped.values()
        for index in indices[:n_samples_per_prompt]
    ]


def build_ttrl_reward_result(
    samples: Sequence[TTRLSample],
    *,
    n_votes_per_prompt: int = 0,
    n_samples_per_prompt: int = 0,
    verifier: TTRLVerifier = DEFAULT_TTRL_VERIFIER,
) -> TTRLRewardResult:
    judge_name = verifier.judge_name
    missing_reason = verifier.missing_reason
    grouped: dict[int, list[tuple[int, TTRLSample, str, str]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        label = verifier.label_for_completion(sample.completion)
        skip_reason = "" if label else missing_reason
        grouped[sample.prompt_idx].append((idx, sample, label, skip_reason))

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
        majority_by_prompt[prompt_idx] = TTRLMajorityResult(
            prompt_idx=prompt_idx,
            label=majority_label,
            ratio=majority_ratio,
            count=best_count,
            total=total,
        )
        for idx, sample, label, skip_reason in group:
            reward = 1.0 if label and label == majority_label else 0.0
            sample_result = TTRLSampleReward(
                prompt_idx=sample.prompt_idx,
                sample_idx=sample.sample_idx,
                completion=sample.completion,
                label=label,
                majority_label=majority_label,
                majority_ratio=majority_ratio,
                reward=reward,
                skip_reason=skip_reason,
            )
            result_by_input_idx[idx] = sample_result

    if n_votes_per_prompt == 0 and n_samples_per_prompt == 0:
        selected_indices = list(range(len(samples)))
    elif n_votes_per_prompt > 0 and n_samples_per_prompt > 0:
        selected_indices = select_ttrl_training_indices(
            samples=samples,
            n_votes_per_prompt=n_votes_per_prompt,
            n_samples_per_prompt=n_samples_per_prompt,
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


def build_ttrl_reward_outcome_batch(
    trace: "GenerationTraceBatch",
    *,
    verifier: TTRLVerifier = DEFAULT_TTRL_VERIFIER,
) -> RewardOutcomeBatch:
    from tllm.runtime.reward_outcome_runtime import RewardOutcomeEntry

    samples = [
        TTRLSample(
            prompt_idx=entry.prompt_idx,
            sample_idx=entry.sample_idx,
            completion=entry.text,
        )
        for entry in trace.entries
    ]
    result = build_ttrl_reward_result(samples, verifier=verifier)
    outcome_entries: list[RewardOutcomeEntry] = []
    for trace_entry, sample_result in zip(trace.entries, result.sample_results, strict=True):
        skipped = sample_result.skip_reason != ""
        outcome_entries.append(
            RewardOutcomeEntry(
                request_id=trace_entry.request_id,
                prompt_idx=trace_entry.prompt_idx,
                sample_idx=trace_entry.sample_idx,
                output_index=trace_entry.output_index,
                group_id=trace_entry.group_id,
                reward=None if skipped else float(sample_result.reward),
                judge=f"ttrl_majority:{result.verifier_name}",
                label="" if skipped else sample_result.label,
                skip_reason=sample_result.skip_reason,
                is_correct=None if skipped else sample_result.reward == 1.0,
            )
        )
    return RewardOutcomeBatch(entries=tuple(outcome_entries))


def _ttrl_barrier_id(group_id: str, prompt_idx: int) -> str:
    return (
        f"ttrl:{group_id}:{prompt_idx}"
        if group_id
        else f"ttrl:prompt:{prompt_idx}"
    )


def build_ttrl_closed_loop_intent(
    trace: "GenerationTraceBatch",
    *,
    consumer_id: str = "ttrl",
    n_votes_per_prompt: int = 0,
    n_samples_per_prompt: int = 0,
    verifier: TTRLVerifier = DEFAULT_TTRL_VERIFIER,
    adapter_version_id: str = "",
) -> TTRLClosedLoopIntent:
    from tllm.runtime.retry_control_runtime import RetryControlBatch, RetryControlEntry
    from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch, UpdateBarrierEntry

    if not consumer_id:
        raise ValueError("consumer_id must be non-empty")
    groups_by_prompt: dict[int, str] = {}
    for entry in trace.entries:
        prior = groups_by_prompt.setdefault(
            entry.prompt_idx, entry.group_id
        )
        if prior != entry.group_id:
            raise ValueError(
                "TTRL requires one generation group per prompt index"
            )
    reward_outcomes = build_ttrl_reward_outcome_batch(trace, verifier=verifier)
    samples = [
        TTRLSample(
            prompt_idx=entry.prompt_idx,
            sample_idx=entry.sample_idx,
            completion=entry.text,
        )
        for entry in trace.entries
    ]
    reward_result = build_ttrl_reward_result(
        samples,
        n_votes_per_prompt=n_votes_per_prompt,
        n_samples_per_prompt=n_samples_per_prompt,
        verifier=verifier,
    )
    selected_train_indices = {
        idx
        for idx in reward_result.selected_indices
        if reward_outcomes.entries[idx].reward is not None
    }
    retry_entries: list[RetryControlEntry] = []
    grouped: dict[tuple[str, int], list["RewardOutcomeEntry"]] = defaultdict(list)
    grouped_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    training_reward_values: list[float] = []
    training_indices: list[int] = []
    for entry_idx, (trace_entry, outcome) in enumerate(zip(trace.entries, reward_outcomes.entries, strict=True)):
        key = (trace_entry.group_id, trace_entry.prompt_idx)
        grouped[key].append(outcome)
        grouped_indices[key].append(entry_idx)
        if outcome.reward is None:
            action = "retry"
            reason = outcome.skip_reason or "missing-reward"
            retry_after_update = False
        elif outcome.is_correct is True:
            action = "accept"
            reason = "majority-answer"
            retry_after_update = False
            if entry_idx in selected_train_indices:
                training_reward_values.append(float(outcome.reward))
                training_indices.append(entry_idx)
        else:
            action = "retry"
            reason = "non-majority-answer"
            retry_after_update = True
            if entry_idx in selected_train_indices:
                training_reward_values.append(float(outcome.reward))
                training_indices.append(entry_idx)
        retry_entries.append(
            RetryControlEntry(
                request_id=outcome.request_id,
                prompt_idx=outcome.prompt_idx,
                sample_idx=outcome.sample_idx,
                output_index=outcome.output_index,
                group_id=outcome.group_id,
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
                consumer_id=consumer_id,
                state="ready" if ready_count else "blocked",
                request_ids=tuple(outcome.request_id for outcome in outcomes),
                ready_count=ready_count,
                reason="majority-reward-ready" if ready_count else "no-trainable-reward",
            )
        )

    retry_schedule = tuple(
        TTRLRetryScheduleEntry(
            request_id=entry.request_id,
            prompt_idx=entry.prompt_idx,
            sample_idx=entry.sample_idx,
            output_index=entry.output_index,
            group_id=entry.group_id,
            adapter_version_id=adapter_version_id,
        )
        for entry in retry_entries
        if entry.retry_after_update
    )

    return TTRLClosedLoopIntent(
        reward_outcomes=reward_outcomes,
        retry_controls=RetryControlBatch(entries=tuple(retry_entries)),
        update_barriers=UpdateBarrierBatch(entries=tuple(barrier_entries)),
        training_row_count=len(training_reward_values),
        selected_reward_count=len(training_reward_values),
        adapter_version_id=adapter_version_id,
        training_indices=tuple(training_indices),
        training_reward_values=tuple(training_reward_values),
        retry_schedule=retry_schedule,
    )


def majority_vote_rewards(
    samples: Sequence[TTRLSample], *, verifier: TTRLVerifier = DEFAULT_TTRL_VERIFIER
) -> list[float]:
    return build_ttrl_reward_result(samples, verifier=verifier).rewards
