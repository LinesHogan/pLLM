#!/usr/bin/env python3
"""Generic append/read/drain storage for typed runtime batches."""

from __future__ import annotations

from typing import Callable, Generic, TypeVar, cast

EntryT = TypeVar("EntryT")
BatchT = TypeVar("BatchT")
StoreT = TypeVar("StoreT")


class AppendStore(Generic[EntryT, BatchT]):
    def __init__(
        self,
        *,
        batch_type: type[BatchT],
        entries_of: Callable[[BatchT], tuple[EntryT, ...]],
        make_batch: Callable[[tuple[EntryT, ...]], BatchT],
    ) -> None:
        self._batch_type = batch_type
        self._entries_of = entries_of
        self._make_batch = make_batch
        self._entries: list[EntryT] = []

    def append(self, batch: BatchT) -> int:
        if not isinstance(batch, self._batch_type):
            raise TypeError(f"batch must be {self._batch_type.__name__}")
        entries = self._entries_of(batch)
        self._entries.extend(entries)
        return len(entries)

    def read(self) -> BatchT:
        return self._make_batch(tuple(self._entries))

    def drain(self) -> BatchT:
        batch = self.read()
        self._entries.clear()
        return batch


class RuntimeAppendSink(Generic[BatchT]):
    def __init__(
        self,
        *,
        runtime: object,
        mode: str,
        port_name: str,
        writer: Callable[[object, BatchT], int],
    ) -> None:
        self.runtime = runtime
        self.mode = mode
        self._port_name = port_name
        self._writer = writer

    def append(self, batch: BatchT) -> int:
        if self.mode != "append":
            raise RuntimeError(f"unsupported {self._port_name} write mode `{self.mode}`")
        return self._writer(self.runtime, batch)

    def __call__(self, batch: BatchT) -> int:
        return self.append(batch)


def runtime_store(
    runtime: object,
    *,
    attribute: str,
    store_type: type[StoreT],
    factory: Callable[[], StoreT],
) -> StoreT:
    value = getattr(runtime, attribute, None)
    if isinstance(value, store_type):
        return cast(StoreT, value)
    store = factory()
    setattr(runtime, attribute, store)
    return store
