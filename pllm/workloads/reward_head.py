"""A reward head fit to majority-vote rewards: the host votes on each prompt's boxed answers (majority_vote), and
RewardHeadWorkload, an ordinary workload trained on the host (pllm.host.make_host), learns the 0/1 rewards from the
completions' hidden rows. RewardHeadFeedback does both as a generate_feedback callable."""

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import torch
from torch.nn import functional as F

import pllm
from pllm.host import (
    BarrierState,
    CompletionInput,
    ControlReason,
    HostTrainer,
    RetryAction,
    RetryControl,
    RewardOutcome,
    UpdateBarrier,
    WeightedRowBatch,
)


def boxed_label(text: str) -> str:
    start = text.rfind("\\boxed")
    if start < 0:
        start = text.rfind("\\fbox")
    left = text.find("{", start) if start >= 0 else -1
    if left < 0:
        return ""
    depth = 0
    for index in range(left, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            answer = text[left + 1 : index]
            nested = re.findall(r"\\boxed\s*\{([^{}]*)\}", answer)
            return re.sub(r"\s+", "", (nested[-1] if nested else answer).lower())
    return ""


@dataclass(frozen=True, slots=True)
class RewardHeadConfig:
    rank: int = 16
    learning_rate: float = 1e-3
    votes_per_prompt: int = 0
    samples_per_prompt: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        if (
            self.rank < 1
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not (
                self.votes_per_prompt == self.samples_per_prompt == 0
                or 0 < self.samples_per_prompt <= self.votes_per_prompt
            )
        ):
            raise ValueError(
                "RewardHeadConfig needs a positive rank and learning rate, and 0 < samples_per_prompt <= votes_per_prompt or both 0"
            )


@dataclass(frozen=True, slots=True)
class MajorityVote:
    """Host feedback for one completion batch, and the batch the reward head learns from."""

    reward_outcomes: tuple[RewardOutcome, ...]
    retry_controls: tuple[RetryControl, ...]
    update_barriers: tuple[UpdateBarrier, ...]
    batch: WeightedRowBatch[torch.Tensor]

    @property
    def update_count(self) -> int:
        """1 when any row trains, else 0: an empty step publishes unchanged values."""
        return int(any(barrier.ready_count for barrier in self.update_barriers))


def majority_vote(value: CompletionInput, config: RewardHeadConfig) -> MajorityVote:
    """Vote each prompt's boxed answers; rows matching the majority train the head toward reward 1."""
    entries = value.completions.entries
    groups: dict[int, list[int]] = {}
    for index, completion in enumerate(entries):
        groups.setdefault(completion.identity.prompt_index, []).append(index)
    labels = tuple(boxed_label(completion.text) for completion in entries)
    majority: dict[int, str] = {}
    selected: list[int] = []
    barriers: list[UpdateBarrier] = []
    for prompt, indices in groups.items():
        group_id = entries[indices[0]].group_id
        if (
            any(entries[i].group_id != group_id for i in indices)
            or config.votes_per_prompt
            and len(indices) != config.votes_per_prompt
        ):
            raise ValueError("majority_vote needs one complete vote group (votes_per_prompt completions) per prompt")
        counts = Counter(labels[i] for i in indices if labels[i])
        majority[prompt] = max(counts, key=lambda label: counts[label]) if counts else ""
        eligible = indices[: config.samples_per_prompt] if config.samples_per_prompt else indices
        trainable = [i for i in eligible if labels[i]] if config.enabled else []
        selected.extend(trainable)
        barriers.append(
            UpdateBarrier(
                f"majority:{group_id}:{prompt}",
                group_id,
                BarrierState.READY if trainable else BarrierState.BLOCKED,
                tuple(entries[i].identity.request_id for i in indices),
                len(trainable),
                ControlReason.REWARD_READY if trainable else ControlReason.NO_REWARD,
            )
        )
    rewards: list[RewardOutcome] = []
    controls: list[RetryControl] = []
    for completion, label in zip(entries, labels, strict=True):
        correct = label == majority[completion.identity.prompt_index] if label else None
        rewards.append(
            RewardOutcome(
                completion,
                float(correct) if correct is not None else None,
                correct,
                "",
                "majority:boxed_normalized_exact",
                label,
                "" if label else ControlReason.MISSING_ANSWER.value,
            )
        )
        controls.append(
            RetryControl(
                completion,
                RetryAction.ACCEPT if correct else RetryAction.RETRY,
                ControlReason.MAJORITY_ANSWER
                if correct
                else ControlReason.NON_MAJORITY_ANSWER
                if label
                else ControlReason.MISSING_ANSWER,
                correct is False,
            )
        )
    hidden = value.hidden
    chosen = set(selected)
    batch = WeightedRowBatch(
        hidden,
        hidden.new_tensor([reward.reward or 0.0 for reward in rewards]),
        hidden.new_tensor([float(index in chosen) for index in range(len(entries))]),
    )
    return MajorityVote(tuple(rewards), tuple(controls), tuple(barriers), batch)


@dataclass(frozen=True)
class RewardHeadFeedback:
    """The `generate_feedback` callable: vote, train the reward head (a RewardHeadWorkload from
    pllm.host.make_host) on the votes, and return the vote."""

    head: HostTrainer
    config: RewardHeadConfig

    def __call__(self, value: CompletionInput) -> MajorityVote:
        vote = majority_vote(value, self.config)
        self.head.train(vote.batch)
        return vote


@dataclass(frozen=True)
class RewardHeadWorkload(pllm.Workload):
    """A rank-r reward head, fit by AdamW to the 0/1 rewards of `majority_vote`."""

    config: RewardHeadConfig = RewardHeadConfig()

    @property
    def head(self) -> pllm.InstanceState:
        rank = self.config.rank
        return pllm.InstanceState(
            {"head": lambda m: (rank * (m.hidden_size + 1),)},
            optimizer=pllm.AdamW(lr=self.config.learning_rate),
            initialize=self.initialize,
        )

    def initialize(self, parameters: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        head = parameters["head"]
        hidden = head.numel() // self.config.rank - 1
        head[: self.config.rank * hidden].normal_(std=hidden**-0.5, generator=generator)

    @pllm.train(state="head")
    def update(
        self,
        hidden: torch.Tensor = pllm.batch("hidden"),
        targets: torch.Tensor = pllm.batch("targets"),
        weights: torch.Tensor = pllm.batch("weights"),
        head: torch.Tensor = pllm.parameter("head", "head"),
    ) -> pllm.UpdateOutput:
        """Weighted mean squared error; an empty selection has zero loss and leaves the head unchanged."""
        rank, width = self.config.rank, hidden.shape[1]
        split = rank * width
        prediction = F.linear(F.linear(hidden, head[:split].view(rank, width)), head[split:].view(1, rank))
        total = weights.sum()
        loss = (weights * (prediction.squeeze(-1) - targets).square()).sum() / torch.where(total > 0, total, 1)
        changed = total > 0
        return pllm.UpdateOutput(loss, changed, changed.to(torch.int64), (weights > 0).sum())
