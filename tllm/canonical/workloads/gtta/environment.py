"""Transactional bounded FIFO for canonical GTTA environment events."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from tllm.canonical.workloads.gtta.types import (
    DynamicsEvent,
    GTTAConfig,
    GTTAHostTransaction,
    GTTATrajectoryBatch,
    TrajectoryFeedback,
    canonical_state_signature,
)


@dataclass(frozen=True, slots=True)
class _NoActiveTransaction:
    pass


_ActiveTransaction = _NoActiveTransaction | GTTAHostTransaction


class GTTAEnvironmentService:
    """Own one bounded event queue and its exact active host transaction."""

    __slots__ = (
        "_active",
        "_config",
        "_dropped_trajectories",
        "_next_transaction_id",
        "_queue",
    )

    def __init__(self, config: GTTAConfig) -> None:
        self._config = config
        self._queue: deque[DynamicsEvent] = deque()
        self._active: _ActiveTransaction = _NoActiveTransaction()
        self._next_transaction_id = 1
        self._dropped_trajectories = 0

    @property
    def config(self) -> GTTAConfig:
        return self._config

    @property
    def dropped_trajectories(self) -> int:
        return self._dropped_trajectories

    @property
    def queued_trajectories(self) -> int:
        return len(self._queue)

    def enqueue_feedback(self, feedback: TrajectoryFeedback) -> bool:
        return self.enqueue_event(
            DynamicsEvent(
                task_id=feedback.task_id.strip(),
                state_signature=canonical_state_signature(
                    feedback.previous_observation
                ),
                action=feedback.action.strip(),
                previous_observation=feedback.previous_observation,
                next_observation=feedback.next_observation,
                environmental_dynamics=feedback.environmental_dynamics,
                reward=feedback.reward,
            )
        )

    def enqueue_event(self, event: DynamicsEvent) -> bool:
        if len(self._queue) >= self._config.max_queue_size:
            self._dropped_trajectories += 1
            return False
        self._queue.append(event)
        return True

    def begin(self, limit: int) -> GTTAHostTransaction:
        match self._active:
            case _NoActiveTransaction():
                pass
            case GTTAHostTransaction():
                raise RuntimeError("GTTA active transaction is unresolved")
        count = len(self._queue) if limit <= 0 else min(limit, len(self._queue))
        transaction = GTTAHostTransaction(
            self._next_transaction_id,
            GTTATrajectoryBatch(
                tuple(self._queue.popleft() for _ in range(count))
            ),
        )
        self._next_transaction_id += 1
        self._active = transaction
        return transaction

    def commit(self, transaction_id: int) -> int:
        transaction = self._active_transaction(transaction_id)
        self._active = _NoActiveTransaction()
        return transaction.transaction_id

    def abort(self, transaction_id: int) -> int:
        transaction = self._active_transaction(transaction_id)
        for event in reversed(transaction.trajectories.entries):
            self._queue.appendleft(event)
        self._active = _NoActiveTransaction()
        return transaction.transaction_id

    def _active_transaction(self, transaction_id: int) -> GTTAHostTransaction:
        if transaction_id < 1:
            raise ValueError("GTTA transaction ID must be positive")
        match self._active:
            case GTTAHostTransaction() as transaction if (
                transaction.transaction_id == transaction_id
            ):
                return transaction
            case _NoActiveTransaction() | GTTAHostTransaction():
                raise RuntimeError("GTTA transaction does not match the active batch")


__all__ = ["GTTAEnvironmentService"]
