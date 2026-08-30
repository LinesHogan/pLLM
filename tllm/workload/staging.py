"""Typed staged ports with workload-instance-owned storage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar, cast

from tllm.workload.runtime_values import SKIPPED, RuntimeValueMarker

T = TypeVar("T")


class StagedLifecycle(StrEnum):
    REQUEST_FINISHED = "request.finished"


@dataclass(frozen=True, slots=True)
class _Empty:
    pass


@dataclass(frozen=True, slots=True)
class _Ready:
    sequence: int
    transaction: str
    value: object


@dataclass(frozen=True, slots=True)
class _Prepared:
    transaction: str
    value: object


def _ordered_transaction(transaction: str) -> tuple[str, int] | None:
    owner, separator, ordinal = transaction.rpartition(":")
    if not separator or not owner or not ordinal.isdecimal():
        return None
    position = int(ordinal)
    if position < 1:
        return None
    return owner, position


@dataclass(frozen=True, slots=True)
class _Finalized:
    ordinal: int
    consumed: bool


class RuntimePortStore:
    """Type-erased storage hidden behind generic StagedPort boundaries."""

    def __init__(self) -> None:
        self._states: dict[str, _Empty | _Prepared | _Ready] = {}
        self._finalized: dict[tuple[str, str], _Finalized] = {}
        self._sequences: dict[str, int] = {}
        self._transaction_ordinals: dict[str, int] = {}

    def stage(self, key: str, value: object) -> int:
        ordinal = self._transaction_ordinals.get(key, 0) + 1
        self._transaction_ordinals[key] = ordinal
        transaction = f"local:{key}:{ordinal}"
        self.prepare(key, transaction, value)
        return self.commit(key, transaction)

    def prepare(self, key: str, transaction: str, value: object) -> None:
        ordered = _ordered_transaction(transaction)
        if ordered is None:
            raise ValueError("staged transaction must be owner:positive_ordinal")
        owner, ordinal = ordered
        finalized = self._finalized.get((key, owner))
        if finalized is not None and ordinal <= finalized.ordinal:
            raise RuntimeError("staged transaction is reused or out of order")
        match self._states.get(key, _Empty()):
            case _Empty():
                self._states[key] = _Prepared(transaction, value)
            case _:
                raise RuntimeError("staged input already contains a pending value")

    def commit(self, key: str, transaction: str) -> int:
        match self._states.get(key, _Empty()):
            case _Prepared(transaction=observed, value=value) if observed == transaction:
                sequence = self._sequences.get(key, 0) + 1
                self._sequences[key] = sequence
                self._states[key] = _Ready(sequence, transaction, value)
                return sequence
            case _:
                raise RuntimeError("staged input transaction is not prepared")

    def abort(self, key: str, transaction: str) -> None:
        match self._states.get(key, _Empty()):
            case _Prepared(transaction=observed) if observed == transaction:
                self._states.pop(key)
                self._record_finalized(key, transaction, consumed=False)
            case _Ready(transaction=observed) if observed == transaction:
                self._states.pop(key)
                sequence = self._sequences[key] - 1
                if sequence:
                    self._sequences[key] = sequence
                else:
                    self._sequences.pop(key)
                self._record_finalized(key, transaction, consumed=False)
            case _Empty():
                ordered = _ordered_transaction(transaction)
                if ordered is None:
                    raise ValueError(
                        "staged transaction must be owner:positive_ordinal"
                    )
                owner, ordinal = ordered
                finalized = self._finalized.get((key, owner))
                if finalized is None or ordinal > finalized.ordinal:
                    self._finalized[(key, owner)] = _Finalized(ordinal, False)
                elif ordinal < finalized.ordinal:
                    raise RuntimeError(
                        "cannot confirm rollback for a superseded transaction"
                    )
                elif finalized.consumed:
                    raise RuntimeError("cannot abort a consumed staged transaction")
            case _:
                raise RuntimeError("cannot abort a different staged transaction")

    def take(self, key: str) -> object | RuntimeValueMarker:
        match self._states.pop(key, _Empty()):
            case _Empty():
                return SKIPPED
            case _Prepared():
                raise RuntimeError("cannot consume an uncommitted staged input")
            case _Ready(transaction=transaction, value=value):
                self._record_finalized(key, transaction, consumed=True)
                return value

    def _record_finalized(
        self, key: str, transaction: str, *, consumed: bool
    ) -> None:
        ordered = _ordered_transaction(transaction)
        if ordered is None:
            raise ValueError("staged transaction must be owner:positive_ordinal")
        owner, ordinal = ordered
        self._finalized[(key, owner)] = _Finalized(ordinal, consumed)


class StagedPort(Generic[T]):
    def __init__(self, *, lifecycle: StagedLifecycle | None = None) -> None:
        self.name = ""
        self.key = ""
        self.owner: type[object] = object
        self.lifecycle = lifecycle

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.name = name
        self.owner = owner
        self.key = f"{owner.__module__}.{owner.__qualname__}:{name}"

    def stage(self, store: RuntimePortStore, value: T) -> int:
        return store.stage(self.key, value)

    def take(
        self, store: RuntimePortStore, _step_id: int
    ) -> T | RuntimeValueMarker:
        value = store.take(self.key)
        if value is SKIPPED:
            return SKIPPED
        return cast(T, value)


__all__ = ["RuntimePortStore", "StagedLifecycle", "StagedPort"]
