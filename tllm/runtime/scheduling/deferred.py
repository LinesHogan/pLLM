#!/usr/bin/env python3
"""Bounded-time queue for timing-movable consumer flow work."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from tllm.contracts.port_bundle import PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.producers.base import HookPoint

if TYPE_CHECKING:
    from tllm.runtime.scheduling.plan_store import PlanResources


WorkRunner = Callable[[PortBundle, RuntimeContext], None]
WorkVerifier = Callable[[PortBundle], None]

_HOOK_INDEX = {hook.value: index for index, hook in enumerate(HookPoint)}


def _timeline_key(step_id: int, hook: str) -> tuple[int, int]:
    try:
        hook_index = _HOOK_INDEX[str(hook)]
    except KeyError as exc:
        raise ValueError(f"unknown deferred-work hook {hook!r}") from exc
    return int(step_id), hook_index


@dataclass(frozen=True, slots=True)
class DeferredWork:
    """One already-materialized flow invocation with a bounded lifetime."""

    flow_id: str
    bundle: PortBundle
    context: RuntimeContext
    run: WorkRunner
    enqueued_step: int
    eligible_step: int
    eligible_hook: str
    deadline_step: int
    deadline_hook: str
    verify: WorkVerifier | None = None
    plan_resources: PlanResources | None = None


class DeferredWorkDecision(StrEnum):
    RUN = "run"
    KEEP = "keep"
    DROP = "drop"


WorkSelector = Callable[[DeferredWork, bool], DeferredWorkDecision]


@dataclass(frozen=True, slots=True)
class DeferredDrainResult:
    ran: int = 0
    dropped: int = 0

    @property
    def resolved(self) -> int:
        return self.ran + self.dropped


class DeferredWorkQueue:
    """FIFO queue drained at compiler-selected bubble/deadline hooks.

    Queue entries intentionally retain their :class:`PortBundle`, including
    tensor payloads. The runtime therefore gives borrowed bundles a deadline no
    later than the next prepare hook, before decode scratch can be reused.
    """

    __slots__ = ("_items", "_max_depth")

    def __init__(self) -> None:
        self._items: deque[DeferredWork] = deque()
        self._max_depth = 0

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[DeferredWork]:
        return iter(self._items)

    @property
    def depth(self) -> int:
        return len(self._items)

    @property
    def max_depth(self) -> int:
        return int(self._max_depth)

    def enqueue(self, work: DeferredWork) -> None:
        if not str(work.flow_id).strip():
            raise ValueError("deferred work requires a non-empty flow_id")
        if int(work.bundle.version) <= 0:
            raise ValueError("deferred work requires a positive bundle version")
        enqueued_step = int(work.enqueued_step)
        eligible = _timeline_key(work.eligible_step, work.eligible_hook)
        deadline = _timeline_key(work.deadline_step, work.deadline_hook)
        if int(work.deadline_step) < enqueued_step:
            raise ValueError("deferred work deadline precedes enqueue step")
        if deadline < eligible:
            raise ValueError("deferred work deadline precedes eligibility")
        self._items.append(work)
        self._max_depth = max(self._max_depth, len(self._items))

    @staticmethod
    def _run(work: DeferredWork, predecessor_events: tuple[object, ...] = ()) -> None:
        if work.verify is not None:
            work.verify(work.bundle)
        context = (
            replace(work.context, predecessor_events=predecessor_events)
            if predecessor_events
            else work.context
        )
        work.run(work.bundle, context)
        DeferredWorkQueue._release_owner(work)

    @staticmethod
    def _release_owner(
        work: DeferredWork,
        *,
        abandon_transfer: bool = False,
    ) -> None:
        resources = work.plan_resources
        if resources is None:
            return
        if abandon_transfer:
            resources.abandon_adaptive_transfer(
                flow_id=work.flow_id,
                bundle_key=work.bundle.key,
                bundle_version=work.bundle.version,
            )
        resources.release_deferred_handoff()

    def drain_on_bubble(
        self,
        *,
        step_id: int,
        hook: str,
        max_items: int | None = None,
        predecessor_events: tuple[object, ...] = (),
    ) -> int:
        """Run ready work in stable order, optionally bounded by a budget.

        Placement order is topological, but independent work may target
        different hooks. A not-yet-ready item therefore cannot head-of-line
        block later independent work that is ready at this bubble.
        """
        if not self._items:
            return 0
        now = _timeline_key(step_id, hook)
        budget = len(self._items) if max_items is None else max(0, int(max_items))
        drained = 0
        pending = self._items
        self._items = deque()
        while pending:
            work = pending.popleft()
            if drained >= budget or _timeline_key(work.eligible_step, work.eligible_hook) > now:
                self._items.append(work)
                continue
            try:
                self._run(work, predecessor_events)
            except Exception:
                self._items.append(work)
                self._items.extend(pending)
                raise
            drained += 1
        return drained

    def flush_expired(
        self,
        *,
        step_id: int,
        hook: str,
        predecessor_events: tuple[object, ...] = (),
    ) -> int:
        """Force expired work without violating topological queue order.

        Ready predecessors before the last expired item are drained as well;
        otherwise an expired successor could run before an unexpired but ready
        predecessor. Not-yet-ready independent items remain queued.
        """
        if not self._items:
            return 0
        now = _timeline_key(step_id, hook)
        last_expired = -1
        for index, work in enumerate(self._items):
            if _timeline_key(work.deadline_step, work.deadline_hook) <= now:
                last_expired = index
        if last_expired < 0:
            return 0

        drained = 0
        pending = self._items
        self._items = deque()
        index = 0
        while pending:
            work = pending.popleft()
            should_run = (
                index <= last_expired
                and _timeline_key(work.eligible_step, work.eligible_hook) <= now
            )
            index += 1
            if not should_run:
                self._items.append(work)
                continue
            try:
                self._run(work, predecessor_events)
            except Exception:
                self._items.append(work)
                self._items.extend(pending)
                raise
            drained += 1
        return drained

    def flush_all(self) -> int:
        """Drain every item during generate-end, synchronize, or shutdown."""
        drained = 0
        while self._items:
            self._run(self._items[0])
            self._items.popleft()
            drained += 1
        return drained

    def drain_selected(
        self,
        *,
        step_id: int,
        hook: str,
        select: WorkSelector,
        expired_only: bool = False,
        predecessor_events: tuple[object, ...] = (),
    ) -> DeferredDrainResult:
        """Resolve ready work through one typed run/keep/drop decision."""

        if not callable(select):
            raise TypeError("deferred work selector must be callable")
        if not self._items:
            return DeferredDrainResult()
        now = _timeline_key(step_id, hook)
        last_expired = len(self._items) - 1
        if expired_only:
            last_expired = -1
            for index, work in enumerate(self._items):
                if _timeline_key(work.deadline_step, work.deadline_hook) <= now:
                    last_expired = index
            if last_expired < 0:
                return DeferredDrainResult()

        ran = 0
        dropped = 0
        pending = self._items
        self._items = deque()
        index = 0
        while pending:
            work = pending.popleft()
            in_scope = (
                index <= last_expired
                and _timeline_key(work.eligible_step, work.eligible_hook) <= now
            )
            index += 1
            if not in_scope:
                self._items.append(work)
                continue
            must_resolve = expired_only or (
                _timeline_key(work.deadline_step, work.deadline_hook) <= now
            )
            try:
                decision = select(work, must_resolve)
            except Exception:
                self._items.append(work)
                self._items.extend(pending)
                raise
            if not isinstance(decision, DeferredWorkDecision):
                self._items.append(work)
                self._items.extend(pending)
                raise TypeError(
                    "deferred work selector must return DeferredWorkDecision"
                )
            if decision == DeferredWorkDecision.KEEP:
                self._items.append(work)
                continue
            if decision == DeferredWorkDecision.DROP:
                self._release_owner(work)
                dropped += 1
                continue
            try:
                self._run(work, predecessor_events)
            except Exception:
                self._items.append(work)
                self._items.extend(pending)
                raise
            ran += 1
        return DeferredDrainResult(ran=ran, dropped=dropped)

    def clear(self) -> None:
        for work in self._items:
            self._release_owner(work, abandon_transfer=True)
        self._items.clear()
