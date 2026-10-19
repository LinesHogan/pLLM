"""Majority reward learning on the original instance arena and execution receipt."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

import torch
from torch.nn import functional as F

import pllm
from pllm.contracts.feedback import (
    BarrierState,
    ControlReason,
    RetryAction,
    RetryControl,
    RewardOutcome,
    UpdateBarrier,
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


@dataclass(slots=True)
class TTRLData:
    batch: pllm.CompletionInput
    reward_outcomes: tuple[RewardOutcome, ...]
    retry_controls: tuple[RetryControl, ...]
    update_barriers: tuple[UpdateBarrier, ...]
    selected_rows: tuple[int, ...]
    loss: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class TTRLResult:
    reward_outcomes: tuple[RewardOutcome, ...]
    retry_controls: tuple[RetryControl, ...]
    update_barriers: tuple[UpdateBarrier, ...]
    loss: torch.Tensor | None
    state_ref: pllm.StateRef
    publication: pllm.PagePublication | None

    @property
    def update_count(self) -> int:
        return int(self.publication is not None)


@dataclass(frozen=True, slots=True)
class TTRLWorkload(pllm.CompletionWorkload[TTRLData, TTRLResult]):
    name = "ttrl"
    config: TTRLConfig = TTRLConfig()

    def training_example(self, facts: pllm.HostRuntimeFacts) -> TTRLData:
        # Reward metadata cannot be inferred from tensor shapes. Select every admitted row
        # so the loss trace covers the maximum training batch, including partial vote groups.
        batch = pllm.CompletionInput.example(facts)
        rewards = tuple(RewardOutcome(entry, 1.0, True, "", "shape", "0", "") for entry in batch.completions.entries)
        return TTRLData(batch, rewards, (), (), tuple(range(len(batch.hidden))))

    @property
    def parameters(self) -> pllm.InstanceState:
        config = self.config

        def initialize(parameters: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
            head = parameters["head"]
            hidden = head.numel() // config.rank - 1
            head[: config.rank * hidden].normal_(std=hidden**-0.5, generator=generator)

        return pllm.InstanceState(
            {"head": lambda m: (config.rank * (m.hidden_size + 1),)},
            namespace="canonical.reward_head@1",
            optimizer=pllm.AdamW(lr=config.learning_rate),
            initialize=initialize,
        )

    def trained_rows(self, data: TTRLData) -> int:
        return len(data.selected_rows)

    def loss(self, data: TTRLData, parameters: Mapping[str, torch.Tensor]) -> torch.Tensor | None:
        if not data.selected_rows:
            return None
        hidden = data.batch.hidden.index_select(0, torch.tensor(data.selected_rows, device=data.batch.hidden.device))
        target = hidden.new_tensor([data.reward_outcomes[i].reward for i in data.selected_rows]).reshape(-1, 1)
        head = parameters["head"]
        split = self.config.rank * hidden.shape[1]
        prediction = F.linear(
            F.linear(hidden, head[:split].view(self.config.rank, hidden.shape[1])),
            head[split:].view(1, self.config.rank),
        )
        loss = F.mse_loss(prediction, target)
        data.loss = loss.detach()
        return loss

    def prepare(self, value: pllm.CompletionInput) -> TTRLData:
        entries = value.completions.entries
        groups: dict[int, list[int]] = {}
        for index, completion in enumerate(entries):
            groups.setdefault(completion.identity.prompt_idx, []).append(index)
        labels = tuple(boxed_label(completion.text) for completion in entries)
        majority: dict[int, str] = {}
        selected: list[int] = []
        barriers: list[UpdateBarrier] = []
        for prompt, indices in groups.items():
            group_id = entries[indices[0]].group_id
            if (
                any(entries[i].group_id != group_id for i in indices)
                or self.config.votes_per_prompt
                and len(indices) != self.config.votes_per_prompt
            ):
                raise ValueError("TTRL requires one complete declared vote group per prompt")
            counts = Counter(labels[i] for i in indices if labels[i])
            majority[prompt] = max(counts, key=lambda label: counts[label]) if counts else ""
            eligible = indices[: self.config.samples_per_prompt] if self.config.samples_per_prompt else indices
            trainable = [i for i in eligible if labels[i]] if self.config.enabled else []
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
            correct = label == majority[completion.identity.prompt_idx] if label else None
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
        return TTRLData(value, tuple(rewards), tuple(controls), tuple(barriers), tuple(selected))

    def output(self, data: TTRLData, result: pllm.InvocationOutput) -> TTRLResult:
        refs = result.publication.after_refs if result.publication is not None else result.selection.refs
        return TTRLResult(
            data.reward_outcomes, data.retry_controls, data.update_barriers, data.loss, refs[0], result.publication
        )
