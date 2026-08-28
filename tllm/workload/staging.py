"""Typed staged ports with workload-instance-owned storage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from tllm.workload.runtime_values import SKIPPED, RuntimeValueMarker

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _Empty:
    pass


@dataclass(frozen=True, slots=True)
class _Ready:
    sequence: int
    value: object


@dataclass(frozen=True, slots=True)
class _Prepared:
    transaction: str
    value: object


class RuntimePortStore:
    """Type-erased storage hidden behind generic StagedPort boundaries."""

    def __init__(self) -> None:
        self._states: dict[str, _Empty | _Prepared | _Ready] = {}
        self._sequence = 0

    def stage(self, key: str, value: object) -> int:
        transaction = f"local:{self._sequence + 1}"
        self.prepare(key, transaction, value)
        return self.commit(key, transaction)

    def prepare(self, key: str, transaction: str, value: object) -> None:
        match self._states.get(key, _Empty()):
            case _Empty():
                self._states[key] = _Prepared(transaction, value)
            case _:
                raise RuntimeError("staged input already contains a pending value")

    def commit(self, key: str, transaction: str) -> int:
        match self._states.get(key, _Empty()):
            case _Prepared(transaction=observed, value=value) if observed == transaction:
                self._sequence += 1
                self._states[key] = _Ready(self._sequence, value)
                return self._sequence
            case _:
                raise RuntimeError("staged input transaction is not prepared")

    def abort(self, key: str, transaction: str) -> None:
        match self._states.get(key, _Empty()):
            case _Prepared(transaction=observed) if observed == transaction:
                self._states.pop(key)
            case _Empty():
                pass
            case _:
                raise RuntimeError("cannot abort a different staged transaction")

    def take(self, key: str) -> object | RuntimeValueMarker:
        match self._states.pop(key, _Empty()):
            case _Empty():
                return SKIPPED
            case _Prepared():
                raise RuntimeError("cannot consume an uncommitted staged input")
            case _Ready(value=value):
                return value


class StagedPort(Generic[T]):
    def __init__(self) -> None:
        self.name = ""
        self.key = ""
        self.owner: type[object] = object

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.name = name
        self.owner = owner
        self.key = f"{owner.__module__}.{owner.__qualname__}:{name}"
        ports = tuple(owner.__dict__.get("__tllm_staged_ports__", ()))
        setattr(owner, "__tllm_staged_ports__", (*ports, self))

    def stage(self, store: RuntimePortStore, value: T) -> int:
        return store.stage(self.key, value)

    def take(
        self, store: RuntimePortStore, _step_id: int
    ) -> T | RuntimeValueMarker:
        value = store.take(self.key)
        if value is SKIPPED:
            return SKIPPED
        return cast(T, value)


__all__ = ["RuntimePortStore", "StagedPort"]
