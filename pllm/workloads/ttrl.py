"""TTRL: majority-vote rewards computed on the host, and a reward head trained on them as an ordinary workload."""

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import torch
from torch.nn import functional as F

import pllm
from pllm.authoring import StepInput
from pllm.execution.state_execution import StateExecution
from pllm.execution.tensor_binding import TensorOutput
from pllm.execution.tensor_frame import TensorFrame
from pllm.host import (
    BarrierState,
    CompletionInput,
    ControlReason,
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
class TTRLConfig:
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
            raise ValueError("TTRL requires positive rank/rate and paired vote/sample counts")


@dataclass(frozen=True, slots=True)
class TTRLResult:
    """Host feedback for one completion batch, and the training step that learns from it."""

    reward_outcomes: tuple[RewardOutcome, ...]
    retry_controls: tuple[RetryControl, ...]
    update_barriers: tuple[UpdateBarrier, ...]
    step: StepInput

    @property
    def update_count(self) -> int:
        """1 when any row trains, else 0: an empty step publishes unchanged values."""
        return int(any(barrier.ready_count for barrier in self.update_barriers))


def ttrl_vote(value: CompletionInput, config: TTRLConfig) -> TTRLResult:
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
            raise ValueError("TTRL requires one complete declared vote group per prompt")
        counts = Counter(labels[i] for i in indices if labels[i])
        majority[prompt] = max(counts, key=lambda label: counts[label]) if counts else ""
        eligible = indices[: config.samples_per_prompt] if config.samples_per_prompt else indices
        trainable = [i for i in eligible if labels[i]] if config.enabled else []
        selected.extend(trainable)
        barriers.append(
            UpdateBarrier(
                f"ttrl:{group_id}:{prompt}",
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
                "ttrl_majority:boxed_normalized_exact",
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
    step = StepInput.of_batch(batch, value.readiness)
    return TTRLResult(tuple(rewards), tuple(controls), tuple(barriers), step)


@dataclass(frozen=True)
class TTRLFeedback:
    """The `generate_feedback` callable: vote, step the reward head (`execution`, a TTRLWorkload compiled with
    pllm.host.make_host) on the votes, and return the result."""

    execution: StateExecution[StepInput, TensorFrame, TensorOutput]
    config: TTRLConfig
    stream: torch.cuda.Stream | None = None

    def __call__(self, value: CompletionInput) -> TTRLResult:
        result = ttrl_vote(value, self.config)
        self.execution.step(result.step, self.stream)
        return result


@dataclass(frozen=True)
class TTRLWorkload(pllm.Workload):
    """A rank-r reward head, fit by AdamW to the 0/1 majority rewards of `ttrl_vote` steps."""

    config: TTRLConfig = TTRLConfig()

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
