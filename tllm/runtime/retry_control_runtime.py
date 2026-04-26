#!/usr/bin/env python3
"""Runtime storage for completed-generation retry control intents."""

from __future__ import annotations

from dataclasses import dataclass

__test__ = False

_RETRY_ACTIONS = frozenset({"retry", "stop", "accept"})


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


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be bool")
    return value


@dataclass(frozen=True)
class RetryControlEntry:
    request_id: str
    prompt_idx: int
    sample_idx: int
    output_index: int = 0
    group_id: str = ""
    action: str = "accept"
    reason: str = ""
    retry_after_update: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _require_str(self.request_id, field_name="request_id"))
        object.__setattr__(self, "prompt_idx", _require_int(self.prompt_idx, field_name="prompt_idx"))
        object.__setattr__(self, "sample_idx", _require_int(self.sample_idx, field_name="sample_idx"))
        object.__setattr__(self, "output_index", _require_int(self.output_index, field_name="output_index"))
        object.__setattr__(self, "group_id", _require_str(self.group_id, field_name="group_id", allow_empty=True))
        action = _require_str(self.action, field_name="action")
        if action not in _RETRY_ACTIONS:
            raise ValueError(f"unsupported retry action: {action}")
        object.__setattr__(self, "action", action)
        object.__setattr__(
            self,
            "reason",
            _require_str(self.reason, field_name="reason", allow_empty=True, preserve_whitespace=True),
        )
        object.__setattr__(
            self,
            "retry_after_update",
            _require_bool(self.retry_after_update, field_name="retry_after_update"),
        )


@dataclass(frozen=True)
class RetryControlBatch:
    entries: tuple[RetryControlEntry, ...]

    def __post_init__(self) -> None:
        entries: list[RetryControlEntry] = []
        for entry in self.entries:
            if not isinstance(entry, RetryControlEntry):
                raise TypeError("RetryControlBatch entries must be RetryControlEntry")
            entries.append(entry)
        object.__setattr__(self, "entries", tuple(entries))


class RetryControlStore:
    def __init__(self) -> None:
        self._entries: list[RetryControlEntry] = []

    def append(self, batch: RetryControlBatch) -> int:
        if not isinstance(batch, RetryControlBatch):
            raise TypeError("batch must be RetryControlBatch")
        self._entries.extend(batch.entries)
        return len(batch.entries)

    def read(self) -> RetryControlBatch:
        return RetryControlBatch(entries=tuple(self._entries))

    def drain(self) -> RetryControlBatch:
        batch = self.read()
        self._entries.clear()
        return batch


def _store_for_runtime(runtime: object) -> RetryControlStore:
    store = getattr(runtime, "retry_control_store", None)
    if isinstance(store, RetryControlStore):
        return store
    store = RetryControlStore()
    setattr(runtime, "retry_control_store", store)
    return store


def write_retry_controls(runtime: object, batch: RetryControlBatch) -> int:
    if not isinstance(batch, RetryControlBatch):
        raise TypeError("batch must be RetryControlBatch")
    setattr(runtime, "last_retry_controls", batch)
    return _store_for_runtime(runtime).append(batch)


def read_retry_controls(runtime: object, *, drain: bool = False) -> RetryControlBatch:
    if not isinstance(drain, bool):
        raise TypeError("drain must be bool")
    store = _store_for_runtime(runtime)
    return store.drain() if drain else store.read()
