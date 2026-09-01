"""Closed values for the clean-room FTTT host-service vertical."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256

import torch

from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.tensor_layout import TensorDType


class FTTTFinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    ERROR = "error"


class FTTTRetryAction(StrEnum):
    ACCEPT = "accept"
    RETRY = "retry"
    STOP = "stop"


class FTTTBarrierState(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"


class FTTTControlReason(StrEnum):
    CORRECT_OUTPUT = "correct_output"
    POSITIVE_FEEDBACK = "positive_feedback"
    INSTANCE_STOPPED = "instance_stopped"
    FEEDBACK_UPDATE = "feedback_update"
    RETRY_LIMIT = "retry_limit"
    FEEDBACK_QUEUE_FULL = "feedback_queue_full"


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
        if type(self.betas) is not tuple or len(self.betas) != 2:
            raise ValueError("FTTT Adam betas require one exact pair")
        beta_one, beta_two = self.betas
        if (
            type(self.rank) is not int
            or self.rank < 1
            or type(self.learning_rate) is not float
            or self.learning_rate <= 0.0
            or type(self.max_queue_size) is not int
            or self.max_queue_size < 0
            or type(self.feedback_feature_size) is not int
            or self.feedback_feature_size < 1
            or type(self.positive_reward_threshold) is not float
            or type(self.max_retries) is not int
            or self.max_retries < 0
            or type(self.edit_scale) is not float
            or self.edit_scale < 0.0
            or type(beta_one) is not float
            or type(beta_two) is not float
            or not 0.0 <= beta_one < 1.0
            or not 0.0 <= beta_two < 1.0
            or type(self.eps) is not float
            or self.eps <= 0.0
        ):
            raise ValueError("FTTT configuration is invalid")

    def execution_parameters_digest(self) -> str:
        fields = (
            "canonical.fttt.parameters@1",
            f"rank={self.rank}",
            f"learning_rate={self.learning_rate.hex()}",
            f"max_queue_size={self.max_queue_size}",
            f"feedback_feature_size={self.feedback_feature_size}",
            f"positive_reward_threshold={self.positive_reward_threshold.hex()}",
            f"max_retries={self.max_retries}",
            f"edit_scale={self.edit_scale.hex()}",
            f"beta_one={self.betas[0].hex()}",
            f"beta_two={self.betas[1].hex()}",
            f"eps={self.eps.hex()}",
        )
        return sha256("\x00".join(fields).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class FTTTSpecialization:
    hidden_size: int
    tensor_dtype: TensorDType
    device: torch.device
    initialization_seed: int

    def __post_init__(self) -> None:
        if (
            type(self.hidden_size) is not int
            or self.hidden_size < 1
            or type(self.tensor_dtype) is not TensorDType
            or self.tensor_dtype is not TensorDType.FLOAT32
            or type(self.device) is not torch.device
            or self.device.type != "cpu"
            or type(self.initialization_seed) is not int
            or self.initialization_seed < 0
        ):
            raise ValueError("the first canonical FTTT slice requires CPU float32")

    def execution_parameters_digest(self, config: FTTTConfig) -> str:
        if type(config) is not FTTTConfig:
            raise TypeError("FTTT execution identity requires exact config")
        fields = (
            "canonical.fttt.execution@1",
            config.execution_parameters_digest(),
            f"hidden_size={self.hidden_size}",
            f"tensor_dtype={self.tensor_dtype.value}",
            "device=cpu",
            f"initialization_seed={self.initialization_seed}",
        )
        return sha256("\x00".join(fields).encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class FTTTCompletion:
    identity: RequestIdentity
    output_index: int
    group_id: str
    token_ids: tuple[int, ...]
    text: str
    finish_reason: FTTTFinishReason

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not RequestIdentity
            or type(self.output_index) is not int
            or self.output_index < 0
            or type(self.group_id) is not str
            or not self.group_id
            or type(self.token_ids) is not tuple
            or any(type(token_id) is not int or token_id < 0 for token_id in self.token_ids)
            or type(self.text) is not str
            or type(self.finish_reason) is not FTTTFinishReason
        ):
            raise ValueError("FTTT completion is not closed")


@dataclass(frozen=True, slots=True)
class FTTTCompletionBatch:
    entries: tuple[FTTTCompletion, ...]

    def __post_init__(self) -> None:
        identities = tuple(entry.identity for entry in self.entries)
        if (
            type(self.entries) is not tuple
            or not self.entries
            or any(type(entry) is not FTTTCompletion for entry in self.entries)
            or len(set(identities)) != len(identities)
        ):
            raise ValueError("FTTT completion batch requires unique exact entries")


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
        if (
            type(self.is_correct) is not bool
            or type(self.reward) is not float
            or type(self.feedback_text) is not str
            or type(self.feedback_token_ids) is not tuple
            or any(
                type(token_id) is not int or token_id < 0
                for token_id in self.feedback_token_ids
            )
            or type(self.label) is not str
            or type(self.judge) is not str
            or not self.judge
            or type(self.skip_reason) is not str
        ):
            raise ValueError("FTTT judge result is not closed")


@dataclass(frozen=True, slots=True)
class FTTTFeedbackRecord:
    completion: FTTTCompletion
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
            type(self.retries) is not int
            or self.retries < 0
            or type(self.stopped) is not bool
            or type(self.adapter_version) is not int
            or self.adapter_version < 0
        ):
            raise ValueError("FTTT instance state is invalid")


@dataclass(frozen=True, slots=True)
class FTTTInstanceKey:
    request_id: str
    prompt_idx: int

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not str
            or not self.request_id
            or type(self.prompt_idx) is not int
            or self.prompt_idx < 0
        ):
            raise ValueError("FTTT instance key is invalid")


@dataclass(frozen=True, slots=True)
class FTTTRewardOutcome:
    completion: FTTTCompletion
    reward: float
    is_correct: bool
    feedback_text: str
    judge: str
    label: str
    skip_reason: str


@dataclass(frozen=True, slots=True)
class FTTTRetryControl:
    completion: FTTTCompletion
    action: FTTTRetryAction
    reason: FTTTControlReason
    retry_after_update: bool


@dataclass(frozen=True, slots=True)
class FTTTUpdateBarrier:
    barrier_id: str
    group_id: str
    state: FTTTBarrierState
    request_ids: tuple[str, ...]
    ready_count: int
    reason: FTTTControlReason


@dataclass(frozen=True, slots=True)
class FTTTLifecycleTransition:
    key: FTTTInstanceKey
    previous: FTTTInstanceState
    next: FTTTInstanceState
    trained: bool


@dataclass(frozen=True, slots=True)
class FTTTClosedLoopIntent:
    transaction_id: int
    reward_outcomes: tuple[FTTTRewardOutcome, ...]
    retry_controls: tuple[FTTTRetryControl, ...]
    update_barriers: tuple[FTTTUpdateBarrier, ...]
    feedback_records: tuple[FTTTFeedbackRecord, ...]
    transitions: tuple[FTTTLifecycleTransition, ...]
    adapter_version: int = 0

    def __post_init__(self) -> None:
        count = len(self.transitions)
        if (
            type(self.transaction_id) is not int
            or self.transaction_id < 1
            or len(self.reward_outcomes) != count
            or len(self.retry_controls) != count
            or len(self.update_barriers) != count
            or type(self.adapter_version) is not int
            or self.adapter_version < 0
        ):
            raise ValueError("FTTT closed-loop intent is not transaction-aligned")


@dataclass(frozen=True, slots=True)
class FTTTTrainingPlan:
    selected_rows: tuple[int, ...]
    intent: FTTTClosedLoopIntent

    def __post_init__(self) -> None:
        if (
            type(self.selected_rows) is not tuple
            or self.selected_rows != tuple(sorted(set(self.selected_rows)))
            or any(row < 0 or row >= len(self.intent.transitions) for row in self.selected_rows)
            or len(self.selected_rows) != len(self.intent.feedback_records)
        ):
            raise ValueError("FTTT training plan is not row-aligned")


@dataclass(frozen=True, slots=True)
class FTTTHostTransaction:
    hidden_rows: torch.Tensor
    plan: FTTTTrainingPlan

    def __post_init__(self) -> None:
        if (
            type(self.hidden_rows) is not torch.Tensor
            or self.hidden_rows.ndim != 2
            or int(self.hidden_rows.shape[0]) != len(self.plan.intent.transitions)
            or type(self.plan) is not FTTTTrainingPlan
        ):
            raise ValueError("FTTT host transaction is not tensor-aligned")


@dataclass(frozen=True, slots=True)
class FTTTNoTransaction:
    pass


FTTTServicePoll = FTTTHostTransaction | FTTTNoTransaction


__all__ = [
    "FTTTBarrierState",
    "FTTTClosedLoopIntent",
    "FTTTCompletion",
    "FTTTCompletionBatch",
    "FTTTConfig",
    "FTTTControlReason",
    "FTTTFeedbackRecord",
    "FTTTFinishReason",
    "FTTTHostTransaction",
    "FTTTInstanceKey",
    "FTTTInstanceState",
    "FTTTJudgeResult",
    "FTTTLifecycleTransition",
    "FTTTNoTransaction",
    "FTTTRewardOutcome",
    "FTTTRetryAction",
    "FTTTRetryControl",
    "FTTTServicePoll",
    "FTTTSpecialization",
    "FTTTTrainingPlan",
    "FTTTUpdateBarrier",
]
