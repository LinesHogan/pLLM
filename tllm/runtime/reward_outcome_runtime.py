#!/usr/bin/env python3
"""Runtime storage for completed-generation reward outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

__test__ = False


def _require_str(value: object, *, field_name: str, allow_empty: bool = False, preserve_whitespace: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must be non-empty")
    if not preserve_whitespace and value != value.strip():
        raise ValueError(f"{field_name} must not contain surrounding whitespace")
    return value


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    return value


def _optional_reward(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("reward must be real")
    return float(value)


def _optional_bool(value: object, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


@dataclass(frozen=True)
class RewardOutcomeEntry:
    request_id: str
    prompt_idx: int
    sample_idx: int
    output_index: int = 0
    group_id: str = ""
    reward: float | None = None
    judge: str = ""
    label: str = ""
    feedback_text: str = ""
    skip_reason: str = ""
    is_correct: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _require_str(self.request_id, field_name="request_id"))
        object.__setattr__(self, "prompt_idx", _require_int(self.prompt_idx, field_name="prompt_idx"))
        object.__setattr__(self, "sample_idx", _require_int(self.sample_idx, field_name="sample_idx"))
        object.__setattr__(self, "output_index", _require_int(self.output_index, field_name="output_index"))
        object.__setattr__(self, "group_id", _require_str(self.group_id, field_name="group_id", allow_empty=True))
        object.__setattr__(self, "reward", _optional_reward(self.reward))
        object.__setattr__(
            self,
            "judge",
            _require_str(self.judge, field_name="judge", allow_empty=True, preserve_whitespace=True),
        )
        object.__setattr__(
            self,
            "label",
            _require_str(self.label, field_name="label", allow_empty=True, preserve_whitespace=True),
        )
        object.__setattr__(
            self,
            "feedback_text",
            _require_str(self.feedback_text, field_name="feedback_text", allow_empty=True, preserve_whitespace=True),
        )
        object.__setattr__(
            self,
            "skip_reason",
            _require_str(self.skip_reason, field_name="skip_reason", allow_empty=True, preserve_whitespace=True),
        )
        object.__setattr__(self, "is_correct", _optional_bool(self.is_correct, field_name="is_correct"))


@dataclass(frozen=True)
class RewardOutcomeBatch:
    entries: tuple[RewardOutcomeEntry, ...]

    def __post_init__(self) -> None:
        entries: list[RewardOutcomeEntry] = []
        for entry in self.entries:
            if not isinstance(entry, RewardOutcomeEntry):
                raise TypeError("RewardOutcomeBatch entries must be RewardOutcomeEntry")
            entries.append(entry)
        object.__setattr__(self, "entries", tuple(entries))


class RewardOutcomeStore:
    def __init__(self) -> None:
        self._entries: list[RewardOutcomeEntry] = []

    def append(self, batch: RewardOutcomeBatch) -> int:
        if not isinstance(batch, RewardOutcomeBatch):
            raise TypeError("batch must be RewardOutcomeBatch")
        self._entries.extend(batch.entries)
        return len(batch.entries)

    def read(self) -> RewardOutcomeBatch:
        return RewardOutcomeBatch(entries=tuple(self._entries))

    def drain(self) -> RewardOutcomeBatch:
        batch = self.read()
        self._entries.clear()
        return batch


def _store_for_runtime(runtime: object) -> RewardOutcomeStore:
    store = getattr(runtime, "reward_outcome_store", None)
    if isinstance(store, RewardOutcomeStore):
        return store
    store = RewardOutcomeStore()
    setattr(runtime, "reward_outcome_store", store)
    return store


def write_reward_outcomes(runtime: object, batch: RewardOutcomeBatch) -> int:
    if not isinstance(batch, RewardOutcomeBatch):
        raise TypeError("batch must be RewardOutcomeBatch")
    setattr(runtime, "last_reward_outcomes", batch)
    return _store_for_runtime(runtime).append(batch)


def read_reward_outcomes(runtime: object, *, drain: bool = False) -> RewardOutcomeBatch:
    if not isinstance(drain, bool):
        raise TypeError("drain must be bool")
    store = _store_for_runtime(runtime)
    return store.drain() if drain else store.read()
