"""FTTT's values: configuration, judge results, queued feedback, the commit intent and the per-step result."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from pllm.authoring import AdamW
from pllm.contracts.request_identity import RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.host import Completion, CompletionInput, RetryControl, RewardOutcome, UpdateBarrier
from pllm.io import StateImport


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

    def state_update(self) -> StateImport[torch.Tensor]:
        """Borrow the original delta rows for a signed single-layer additive patch."""
        return StateImport(
            (self.edit_delta,),
            tuple(
                RequestStateKey.project(record.completion.identity, (RequestKeyField.PROMPT_INDEX,))
                for record in self.intent.feedback_records
            ),
        )


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

    @property
    def feedback_width(self) -> int:
        """The width of each failed row's feedback: its hashed features and its reward."""
        return self.feedback_feature_size + 1

    def __post_init__(self) -> None:
        AdamW(self.learning_rate, self.betas, self.eps)
        if (
            self.rank < 1
            or self.feedback_feature_size < 1
            or self.max_queue_size < 0
            or self.max_retries < 0
            or not isfinite(self.edit_scale)
            or self.edit_scale < 0.0
            or not isfinite(self.positive_reward_threshold)
        ):
            raise ValueError(
                "FTTTConfig needs positive rank/feedback_feature_size, nonnegative max_queue_size/max_retries/edit_scale, and finite thresholds"
            )


@dataclass(frozen=True, slots=True)
class FTTTJudgeResult:
    is_correct: bool
    reward: float
    feedback_text: str
    feedback_token_ids: tuple[int, ...]
    label: str
    judge: str
    skip_reason: str


@dataclass(frozen=True, slots=True)
class FTTTFeedbackRecord:
    completion: Completion
    feedback_text: str
    feedback_token_ids: tuple[int, ...]
    reward: float


@dataclass(frozen=True, slots=True)
class FTTTRequestLifecycle:
    retries: int = 0
    stopped: bool = False
    adapter_version: int = 0


@dataclass(frozen=True, slots=True)
class FTTTLifecycleTransition:
    key: tuple[str, int]
    previous: FTTTRequestLifecycle
    next: FTTTRequestLifecycle
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


@dataclass(frozen=True, slots=True)
class FTTTQueuedFeedback:
    """One judged completion batch waiting for its edit step: the rows to train and what settles afterwards."""

    batch: CompletionInput
    selected_rows: tuple[int, ...]
    intent: FTTTClosedLoopIntent


__all__ = [
    "FTTTClosedLoopIntent",
    "FTTTConfig",
    "FTTTFeedbackRecord",
    "FTTTQueuedFeedback",
    "FTTTRequestLifecycle",
    "FTTTJudgeResult",
    "FTTTLifecycleTransition",
]
