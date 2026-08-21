#!/usr/bin/env python3
"""Typed stores for prompt context and trajectory dynamics (port-runtime backing)."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Sequence

__test__ = False

_PROMPT_CONTEXT_POSITIONS = frozenset({"prefix", "suffix"})


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


@dataclass(frozen=True)
class TrajectoryEvent:
    task_id: str
    state_signature: str
    previous_observation: str
    action: str
    next_observation: str
    reward: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _require_str(self.task_id, field_name="task_id"))
        object.__setattr__(
            self,
            "state_signature",
            _require_str(self.state_signature, field_name="state_signature"),
        )
        object.__setattr__(
            self,
            "previous_observation",
            _require_str(
                self.previous_observation,
                field_name="previous_observation",
                allow_empty=True,
                preserve_whitespace=True,
            ),
        )
        object.__setattr__(
            self,
            "action",
            _require_str(self.action, field_name="action"),
        )
        object.__setattr__(
            self,
            "next_observation",
            _require_str(
                self.next_observation,
                field_name="next_observation",
                allow_empty=True,
                preserve_whitespace=True,
            ),
        )
        object.__setattr__(self, "reward", _optional_reward(self.reward))


@dataclass(frozen=True)
class TrajectoryEventBatch:
    entries: tuple[TrajectoryEvent, ...]

    def __post_init__(self) -> None:
        entries: list[TrajectoryEvent] = []
        for entry in self.entries:
            if not isinstance(entry, TrajectoryEvent):
                raise TypeError("TrajectoryEventBatch entries must be TrajectoryEvent")
            entries.append(entry)
        object.__setattr__(self, "entries", tuple(entries))


@dataclass(frozen=True)
class PromptContextEntry:
    prompt_idx: int
    text: str
    position: str = "prefix"

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_idx", _require_int(self.prompt_idx, field_name="prompt_idx"))
        object.__setattr__(
            self,
            "text",
            _require_str(self.text, field_name="text", allow_empty=True, preserve_whitespace=True),
        )
        position = _require_str(self.position, field_name="position")
        if position not in _PROMPT_CONTEXT_POSITIONS:
            raise ValueError(f"unsupported prompt context position: {position}")
        object.__setattr__(self, "position", position)


@dataclass(frozen=True)
class PromptContextBatch:
    entries: tuple[PromptContextEntry, ...]

    def __post_init__(self) -> None:
        entries: list[PromptContextEntry] = []
        for entry in self.entries:
            if not isinstance(entry, PromptContextEntry):
                raise TypeError("PromptContextBatch entries must be PromptContextEntry")
            entries.append(entry)
        object.__setattr__(self, "entries", tuple(entries))


class DynamicsStore:
    def __init__(self) -> None:
        self._events: list[TrajectoryEvent] = []

    def append(self, batch: TrajectoryEventBatch) -> int:
        if not isinstance(batch, TrajectoryEventBatch):
            raise TypeError("batch must be TrajectoryEventBatch")
        self._events.extend(batch.entries)
        return len(batch.entries)

    def read(self) -> TrajectoryEventBatch:
        return TrajectoryEventBatch(entries=tuple(self._events))

    def query(self, *, task_id: str = "", state_signature: str = "", action: str = "") -> tuple[TrajectoryEvent, ...]:
        task_filter = _require_str(task_id, field_name="task_id", allow_empty=True)
        state_filter = _require_str(state_signature, field_name="state_signature", allow_empty=True)
        action_filter = _require_str(action, field_name="action", allow_empty=True)
        return tuple(
            event
            for event in self._events
            if (not task_filter or event.task_id == task_filter)
            and (not state_filter or event.state_signature == state_filter)
            and (not action_filter or event.action == action_filter)
        )


class PromptContextStore:
    def __init__(self) -> None:
        self._entries: list[PromptContextEntry] = []

    def append(self, batch: PromptContextBatch) -> int:
        if not isinstance(batch, PromptContextBatch):
            raise TypeError("batch must be PromptContextBatch")
        self._entries.extend(batch.entries)
        return len(batch.entries)

    def read(self) -> PromptContextBatch:
        return PromptContextBatch(entries=tuple(self._entries))

    def entries_for_prompt(self, prompt_idx: int) -> tuple[PromptContextEntry, ...]:
        prompt_idx_i = _require_int(prompt_idx, field_name="prompt_idx")
        return tuple(entry for entry in self._entries if entry.prompt_idx == prompt_idx_i)


def _dynamics_store_for_runtime(runtime: object) -> DynamicsStore:
    store = getattr(runtime, "dynamics_store", None)
    if isinstance(store, DynamicsStore):
        return store
    store = DynamicsStore()
    setattr(runtime, "dynamics_store", store)
    return store


def _prompt_context_store_for_runtime(runtime: object) -> PromptContextStore:
    store = getattr(runtime, "prompt_context_store", None)
    if isinstance(store, PromptContextStore):
        return store
    store = PromptContextStore()
    setattr(runtime, "prompt_context_store", store)
    return store




def write_prompt_contexts(runtime: object, batch: PromptContextBatch) -> int:
    if not isinstance(batch, PromptContextBatch):
        raise TypeError("batch must be PromptContextBatch")
    setattr(runtime, "last_prompt_contexts", batch)
    return _prompt_context_store_for_runtime(runtime).append(batch)



def _prompt_idx_for_row(row_i: int, request_prompt_indices: Sequence[int] | None) -> int:
    if request_prompt_indices is None:
        return int(row_i)
    return _require_int(request_prompt_indices[row_i], field_name="request_prompt_indices entry")


def apply_prompt_contexts(
    runtime: object,
    prompts: Sequence[str],
    *,
    request_prompt_indices: Sequence[int] | None = None,
) -> tuple[str, ...]:
    store = getattr(runtime, "prompt_context_store", None)
    if not isinstance(store, PromptContextStore):
        return tuple(str(prompt) for prompt in prompts)

    out: list[str] = []
    for row_i, prompt in enumerate(prompts):
        prompt_idx = _prompt_idx_for_row(int(row_i), request_prompt_indices)
        text = str(prompt)
        for entry in store.entries_for_prompt(prompt_idx):
            if not entry.text:
                continue
            if entry.position == "prefix":
                text = f"{entry.text}\n{text}"
            else:
                text = f"{text}\n{entry.text}"
        out.append(text)
    return tuple(out)
