#!/usr/bin/env python3
"""Runtime storage for completed-generation update barriers."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.runtime.append_store import AppendStore, runtime_store

__test__ = False

_BARRIER_STATES = frozenset({"pending", "ready", "blocked"})


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


def _request_ids_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, tuple):
        raise TypeError("request_ids must be tuple")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("request_ids entries must be str")
        if not item:
            raise ValueError("request_ids entries must be non-empty")
        if item != item.strip():
            raise ValueError("request_ids entries must not contain surrounding whitespace")
        out.append(item)
    return tuple(out)


@dataclass(frozen=True)
class UpdateBarrierEntry:
    barrier_id: str
    group_id: str
    consumer_id: str
    state: str = "pending"
    request_ids: tuple[str, ...] = ()
    ready_count: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "barrier_id", _require_str(self.barrier_id, field_name="barrier_id"))
        object.__setattr__(self, "group_id", _require_str(self.group_id, field_name="group_id", allow_empty=True))
        object.__setattr__(self, "consumer_id", _require_str(self.consumer_id, field_name="consumer_id"))
        state = _require_str(self.state, field_name="state")
        if state not in _BARRIER_STATES:
            raise ValueError(f"unsupported barrier state: {state}")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "request_ids", _request_ids_tuple(self.request_ids))
        object.__setattr__(self, "ready_count", _require_int(self.ready_count, field_name="ready_count"))
        object.__setattr__(
            self,
            "reason",
            _require_str(self.reason, field_name="reason", allow_empty=True, preserve_whitespace=True),
        )


@dataclass(frozen=True)
class UpdateBarrierBatch:
    entries: tuple[UpdateBarrierEntry, ...]

    def __post_init__(self) -> None:
        entries: list[UpdateBarrierEntry] = []
        for entry in self.entries:
            if not isinstance(entry, UpdateBarrierEntry):
                raise TypeError("UpdateBarrierBatch entries must be UpdateBarrierEntry")
            entries.append(entry)
        object.__setattr__(self, "entries", tuple(entries))


class UpdateBarrierStore(AppendStore[UpdateBarrierEntry, UpdateBarrierBatch]):
    def __init__(self) -> None:
        super().__init__(
            batch_type=UpdateBarrierBatch,
            entries_of=lambda batch: batch.entries,
            make_batch=lambda entries: UpdateBarrierBatch(entries=entries),
        )


def _store_for_runtime(runtime: object) -> UpdateBarrierStore:
    return runtime_store(
        runtime,
        attribute="update_barrier_store",
        store_type=UpdateBarrierStore,
        factory=UpdateBarrierStore,
    )


def write_update_barriers(runtime: object, batch: UpdateBarrierBatch) -> int:
    if not isinstance(batch, UpdateBarrierBatch):
        raise TypeError("batch must be UpdateBarrierBatch")
    setattr(runtime, "last_update_barriers", batch)
    return _store_for_runtime(runtime).append(batch)


def read_update_barriers(runtime: object, *, drain: bool = False) -> UpdateBarrierBatch:
    if not isinstance(drain, bool):
        raise TypeError("drain must be bool")
    store = _store_for_runtime(runtime)
    return store.drain() if drain else store.read()
