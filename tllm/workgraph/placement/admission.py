#!/usr/bin/env python3
"""Copy admission and move execution contracts for WorkGraph placement."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable
from tllm.workgraph.placement.transfer import Move
from tllm.workgraph.placement.vocab import (
    require_instance,
    require_int,
    require_str,
    require_tuple_of,
)


@dataclass(frozen=True, slots=True)
class Lease:
    key: str
    live_start: int
    live_end: int
    ready: bool
    retired: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", require_str(self.key, "Lease.key"))
        start = require_int(self.live_start, "Lease.live_start")
        end = require_int(self.live_end, "Lease.live_end")
        if end < start:
            raise ValueError("Lease.live_end must be >= live_start")
        if not isinstance(cast(object, self.ready), bool) or not isinstance(
            cast(object, self.retired), bool
        ):
            raise TypeError("Lease ready/retired flags must be bool")
@dataclass(frozen=True, slots=True)
class CopyAdmission:
    source: str
    destination: str
    at_step: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", require_str(self.source, "CopyAdmission.source"))
        object.__setattr__(
            self,
            "destination",
            require_str(self.destination, "CopyAdmission.destination"),
        )
        object.__setattr__(
            self,
            "at_step",
            require_int(self.at_step, "CopyAdmission.at_step"),
        )
def admit_reuse(lease: Lease, *, live_start: int, live_end: int) -> bool:
    lease = require_instance(lease, Lease, "admit_reuse.lease")
    start = require_int(live_start, "admit_reuse.live_start")
    end = require_int(live_end, "admit_reuse.live_end")
    if end < start:
        raise ValueError("reuse live_end must be >= live_start")
    return lease.retired or end < lease.live_start or start > lease.live_end
def admit_copy(copy: CopyAdmission, *, leases: tuple[Lease, ...]) -> bool:
    copy = require_instance(copy, CopyAdmission, "admit_copy.copy")
    leases = require_tuple_of(leases, Lease, "admit_copy.leases")
    return all(
        lease.key != copy.destination
        or lease.retired
        or not (lease.live_start <= copy.at_step <= lease.live_end)
        for lease in leases
    )
class CopyEngineAdmission:
    """Bounded copy credits released only by retired leases."""

    __slots__ = ("_capacity", "_active")

    def __init__(self, capacity: int = 1) -> None:
        self._capacity = require_int(capacity, "CopyEngineAdmission.capacity", minimum=1)
        self._active: dict[str, Lease] = {}

    def admit(self, copy: CopyAdmission, *, leases: tuple[Lease, ...]) -> bool:
        self.reap(leases)
        if len(self._active) >= self._capacity or not admit_copy(copy, leases=leases):
            return False
        matching = next(
            (lease for lease in leases if lease.key == copy.destination),
            None,
        )
        if matching is None:
            return False
        self._active[matching.key] = matching
        return True

    def reap(self, leases: tuple[Lease, ...]) -> None:
        current = {lease.key: lease for lease in leases}
        self._active = {
            key: lease
            for key, lease in self._active.items()
            if key in current and not current[key].retired
        }
@runtime_checkable
class CopyRuntime(Protocol):
    def copy(self, kind: str, source: str, destination: str) -> None: ...
def execute_moves(moves: Iterable[Move], *, runtime: CopyRuntime) -> None:
    if not isinstance(cast(object, runtime), CopyRuntime):
        raise TypeError("execute_moves runtime must implement copy")
    for move in moves:
        move = require_instance(move, Move, "execute_moves.move")
        runtime.copy(
            move.kind.value,
            str(move.source.placement),
            str(move.destination.placement),
        )
