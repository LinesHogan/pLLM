"""GTTA's bounded FIFO of environment events waiting to be folded into the dynamics memory."""

from __future__ import annotations

from collections import deque

from pllm.workloads.gtta.types import (
    DynamicsEvent,
    GTTAConfig,
    GTTATrajectoryBatch,
    TrajectoryFeedback,
    canonical_state_signature,
)


class GTTAEnvironment:
    """One bounded event queue; events past max_queue_size are dropped and counted."""

    __slots__ = ("_config", "_dropped_trajectories", "_queue")

    def __init__(self, config: GTTAConfig) -> None:
        self._config = config
        self._queue: deque[DynamicsEvent] = deque()
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
        if len(self._queue) >= self._config.max_queue_size:
            self._dropped_trajectories += 1
            return False
        self._queue.append(
            DynamicsEvent(
                task_id=feedback.task_id.strip(),
                state_signature=canonical_state_signature(feedback.previous_observation),
                action=feedback.action.strip(),
                previous_observation=feedback.previous_observation,
                next_observation=feedback.next_observation,
                environmental_dynamics=feedback.environmental_dynamics,
                reward=feedback.reward,
            )
        )
        return True

    def take(self, limit: int) -> GTTATrajectoryBatch:
        """Remove the oldest `limit` queued events (all of them for 0)."""
        count = len(self._queue) if limit == 0 else min(limit, len(self._queue))
        return GTTATrajectoryBatch(tuple(self._queue.popleft() for _ in range(count)))

    def requeue(self, trajectories: GTTATrajectoryBatch) -> None:
        """Put events taken by a failed run back at the front, in their order."""
        self._queue.extendleft(reversed(trajectories.entries))


__all__ = ["GTTAEnvironment"]
