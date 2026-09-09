"""Closed values for the clean-room FTTT host-service vertical."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from pllm.execution.host_runtime import CompletionInput
from pllm.authoring import AdamW, StateTensorUpdate
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.feedback import Completion, RewardOutcome, RetryControl, UpdateBarrier


@dataclass(frozen=True, slots=True)
class FTTTResult:
    transaction_id: int
    edit_delta: torch.Tensor
    intent: FTTTClosedLoopIntent
    losses: tuple[torch.Tensor, ...]
    update_count: int
    state_version: int

    @property
    def retry_controls(self) -> tuple[RetryControl, ...]:
        return self.intent.retry_controls

    @property
    def update_barriers(self) -> tuple[UpdateBarrier, ...]:
        return self.intent.update_barriers

    def state_update(self) -> StateTensorUpdate[torch.Tensor]:
        """Borrow the original delta rows for a signed single-layer additive patch."""
        return StateTensorUpdate((self.edit_delta,), tuple(RequestStateKey.project(record.completion.identity,
            (RequestKeyField.PROMPT_IDX,)) for record in self.intent.feedback_records))


@dataclass(frozen=True, slots=True)
class FTTTConfig:
    rank: int = 16
    learning_rate: float = 1e-3
    max_queue_size: int = 128
    feedback_feature_size: int = 32
    positive_reward_threshold: float = 1.0
    max_retries: int = 0
    edit_scale: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        AdamW(self.learning_rate, self.betas, self.eps)
        if (self.rank < 1 or self.feedback_feature_size < 1 or self.max_queue_size < 0 or self.max_retries < 0
            or not isfinite(self.edit_scale) or self.edit_scale < 0.0 or not isfinite(self.positive_reward_threshold)):
            raise ValueError("FTTT configuration is invalid")

@dataclass(frozen=True, slots=True)
class FTTTJudgeResult:
    is_correct: bool
    reward: float
    feedback_text: str
    feedback_token_ids: tuple[int, ...]
    label: str
    judge: str
    skip_reason: str

    def __post_init__(self) -> None:
        if any(token_id < 0 for token_id in self.feedback_token_ids) or not self.judge:
            raise ValueError("FTTT judge result is not closed")


@dataclass(frozen=True, slots=True)
class FTTTFeedbackRecord:
    completion: Completion
    feedback_text: str
    feedback_token_ids: tuple[int, ...]
    reward: float


@dataclass(frozen=True, slots=True)
class FTTTInstanceState:
    retries: int = 0
    stopped: bool = False
    adapter_version: int = 0

    def __post_init__(self) -> None:
        if (
            self.retries < 0
            or self.adapter_version < 0
        ):
            raise ValueError("FTTT instance state is invalid")


@dataclass(frozen=True, slots=True)
class FTTTLifecycleTransition:
    key: tuple[str, int]
    previous: FTTTInstanceState
    next: FTTTInstanceState
    trained: bool


@dataclass(frozen=True, slots=True)
class FTTTClosedLoopIntent:
    transaction_id: int
    reward_outcomes: tuple[RewardOutcome, ...]
    retry_controls: tuple[RetryControl, ...]
    update_barriers: tuple[UpdateBarrier, ...]
    feedback_records: tuple[FTTTFeedbackRecord, ...]
    transitions: tuple[FTTTLifecycleTransition, ...]
    adapter_version: int = 0

    def __post_init__(self) -> None:
        count = len(self.transitions)
        if (
            self.transaction_id < 1
            or len(self.reward_outcomes) != count
            or len(self.retry_controls) != count
            or len(self.update_barriers) != count
            or self.adapter_version < 0
        ):
            raise ValueError("FTTT closed-loop intent is not transaction-aligned")


@dataclass(frozen=True, slots=True)
class FTTTHostTransaction:
    batch: CompletionInput
    selected_rows: tuple[int, ...]
    intent: FTTTClosedLoopIntent

    def __post_init__(self) -> None:
        if (
            len(self.intent.transitions) != len(self.batch.completions.entries)
            or self.selected_rows != tuple(sorted(set(self.selected_rows)))
            or any(row < 0 or row >= len(self.intent.transitions) for row in self.selected_rows)
            or len(self.selected_rows) != len(self.intent.feedback_records)
        ):
            raise ValueError("FTTT training plan is not row-aligned")


__all__ = [
    "FTTTClosedLoopIntent",
    "FTTTConfig",
    "FTTTFeedbackRecord",
    "FTTTHostTransaction",
    "FTTTInstanceState",
    "FTTTJudgeResult",
    "FTTTLifecycleTransition",
]
