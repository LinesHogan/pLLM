"""Closed values for the canonical GTTA environment-service vertical."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class GTTAResult:
    transaction_id: int
    update: GTTAUpdateReceipt
    query: GTTAQueryResult


class GTTAPromptPosition(StrEnum):
    PREFIX = "prefix"
    SUFFIX = "suffix"


@dataclass(frozen=True, slots=True)
class GTTAConfig:
    max_queue_size: int = 128
    score_momentum: float = 0.0

    def __post_init__(self) -> None:
        if self.max_queue_size < 0 or not 0.0 <= self.score_momentum <= 1.0:
            raise ValueError("GTTA configuration values are invalid")


def canonical_state_signature(observation: str) -> str:
    return " ".join(observation.split()).casefold()


@dataclass(frozen=True, slots=True)
class TrajectoryFeedback:
    task_id: str
    action: str
    previous_observation: str
    next_observation: str
    environmental_dynamics: str
    reward: float

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.action.strip():
            raise ValueError("GTTA feedback requires a task and action")


@dataclass(frozen=True, slots=True)
class DynamicsEvent:
    task_id: str
    state_signature: str
    action: str
    previous_observation: str
    next_observation: str
    environmental_dynamics: str
    reward: float

    def __post_init__(self) -> None:
        if (
            not self.task_id
            or self.task_id != self.task_id.strip()
            or not self.action
            or self.action != self.action.strip()
        ):
            raise ValueError("GTTA dynamics event identity is not canonical")


@dataclass(frozen=True, slots=True)
class GTTATrajectoryBatch:
    entries: tuple[DynamicsEvent, ...]


@dataclass(frozen=True, slots=True)
class GTTAQuery:
    prompt_idx: int
    task_id: str
    candidate_actions: tuple[str, ...]
    observation: str
    state_signature: str
    action: str
    max_records: int
    position: GTTAPromptPosition

    def __post_init__(self) -> None:
        if self.prompt_idx < 0 or self.max_records < 0:
            raise ValueError("GTTA query bounds are invalid")


@dataclass(frozen=True, slots=True)
class PromptContext:
    prompt_idx: int
    text: str
    position: GTTAPromptPosition

    def __post_init__(self) -> None:
        if self.prompt_idx < 0:
            raise ValueError("GTTA prompt index is invalid")


@dataclass(frozen=True, slots=True)
class GTTAUpdateReceipt:
    processed: int
    state_version: int

    def __post_init__(self) -> None:
        if self.processed < 0 or self.state_version < 0:
            raise ValueError("GTTA update receipt is not closed")


@dataclass(frozen=True, slots=True)
class GTTAQueryResult:
    ranked_actions: tuple[tuple[str, float], ...]
    records: tuple[DynamicsEvent, ...]
    prompt_context: PromptContext


@dataclass(frozen=True, slots=True)
class GTTAHostRequest:
    query: GTTAQuery
    limit: int = 0

    def __post_init__(self) -> None:
        if self.limit < 0:
            raise ValueError("GTTA host request limit must be non-negative")


@dataclass(frozen=True, slots=True)
class GTTAHostTransaction:
    transaction_id: int
    trajectories: GTTATrajectoryBatch
    query: GTTAQuery

    def __post_init__(self) -> None:
        if self.transaction_id < 1:
            raise ValueError("GTTA host transaction is not closed")


__all__ = [
    "DynamicsEvent",
    "GTTAConfig",
    "GTTAHostRequest",
    "GTTAHostTransaction",
    "GTTAPromptPosition",
    "GTTAQuery",
    "GTTAQueryResult",
    "GTTATrajectoryBatch",
    "GTTAUpdateReceipt",
    "PromptContext",
    "TrajectoryFeedback",
    "canonical_state_signature",
]
