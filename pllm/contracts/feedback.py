"""Shared completion, reward and retry boundary; identities remain original references."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pllm.contracts.request_identity import RequestIdentity


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"


class RetryAction(StrEnum):
    ACCEPT = "accept"
    RETRY = "retry"
    STOP = "stop"


class BarrierState(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"


class ControlReason(StrEnum):
    CORRECT_OUTPUT = "correct_output"
    POSITIVE_FEEDBACK = "positive_feedback"
    INSTANCE_STOPPED = "instance_stopped"
    FEEDBACK_UPDATE = "feedback_update"
    RETRY_LIMIT = "retry_limit"
    FEEDBACK_QUEUE_FULL = "feedback_queue_full"
    MAJORITY_ANSWER = "majority-answer"
    NON_MAJORITY_ANSWER = "non-majority-answer"
    MISSING_ANSWER = "missing-boxed-answer"
    REWARD_READY = "majority-reward-ready"
    NO_REWARD = "no-trainable-reward"


@dataclass(frozen=True, slots=True)
class Completion:
    identity: RequestIdentity
    output_index: int
    group_id: str
    token_ids: tuple[int, ...]
    text: str
    finish_reason: FinishReason

    def __post_init__(self) -> None:
        if self.output_index < 0 or not self.group_id or any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("completion is not closed")


@dataclass(frozen=True, slots=True)
class CompletionBatch:
    entries: tuple[Completion, ...]

    def __post_init__(self) -> None:
        identities = tuple(entry.identity for entry in self.entries)
        if not self.entries or len(set(identities)) != len(identities):
            raise ValueError("completion batch requires unique exact entries")


@dataclass(frozen=True, slots=True)
class RewardOutcome:
    completion: Completion
    reward: float | None
    is_correct: bool | None
    feedback_text: str
    judge: str
    label: str
    skip_reason: str


@dataclass(frozen=True, slots=True)
class RetryControl:
    completion: Completion
    action: RetryAction
    reason: ControlReason
    retry_after_update: bool


@dataclass(frozen=True, slots=True)
class UpdateBarrier:
    barrier_id: str
    group_id: str
    state: BarrierState
    request_ids: tuple[str, ...]
    ready_count: int
    reason: ControlReason


class CompletionFeedback(Protocol):
    """Borrow the original settled control values; no second feedback envelope."""

    @property
    def retry_controls(self) -> tuple[RetryControl, ...]: ...

    @property
    def update_barriers(self) -> tuple[UpdateBarrier, ...]: ...

    @property
    def update_count(self) -> int: ...
