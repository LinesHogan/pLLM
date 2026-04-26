#!/usr/bin/env python3
"""GTTA-inspired trajectory feedback consumer."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow


def parse_reward(value: float | int | str | Mapping[str, Any]) -> float:
    if isinstance(value, Mapping):
        if "reward" not in value:
            raise ValueError("reward mapping must contain a 'reward' key")
        return parse_reward(value["reward"])
    return float(value)


@dataclass(frozen=True)
class TrajectoryFeedback:
    task_id: str
    action: str
    observation: str
    reward: float | int | str | Mapping[str, Any]

    def __post_init__(self) -> None:
        if not str(self.task_id).strip():
            raise ValueError("TrajectoryFeedback requires task_id")
        if not str(self.action).strip():
            raise ValueError("TrajectoryFeedback requires action")
        parse_reward(self.reward)

    @property
    def reward_value(self) -> float:
        return parse_reward(self.reward)


@dataclass
class GTTAConsumerConfig:
    consumer_id: str = "gtta"
    enabled: bool = True
    max_queue_size: int = 128
    score_momentum: float = 0.0


@dataclass(frozen=True)
class GTTAStats:
    queued_trajectories: int = 0
    dropped_trajectories: int = 0
    processed_trajectories: int = 0
    update_count: int = 0


class GTTAConsumer(BaseConsumer):
    """Local action-score updater for environment trajectory feedback."""

    def __init__(self, config: GTTAConsumerConfig) -> None:
        self.config = config
        self._queue: deque[TrajectoryFeedback] = deque()
        self._dropped_trajectories = 0
        self._processed_trajectories = 0
        self._update_count = 0
        self._action_scores: dict[str, float] = defaultdict(float)
        self._action_counts: dict[str, int] = defaultdict(int)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        return ()

    def enqueue_trajectory(self, feedback: TrajectoryFeedback) -> bool:
        if len(self._queue) >= max(0, int(self.config.max_queue_size)):
            self._dropped_trajectories += 1
            return False
        self._queue.append(feedback)
        return True

    def process_queued_trajectories(self, *, limit: int = 0) -> None:
        max_items = len(self._queue) if int(limit) <= 0 else min(int(limit), len(self._queue))
        for _ in range(max_items):
            feedback = self._queue.popleft()
            if not bool(self.config.enabled):
                continue
            action = str(feedback.action).strip()
            reward = float(feedback.reward_value)
            old = float(self._action_scores[action])
            momentum = min(1.0, max(0.0, float(self.config.score_momentum)))
            new_score = old * momentum + reward * (1.0 - momentum)
            self._action_scores[action] = new_score
            self._action_counts[action] += 1
            self._processed_trajectories += 1
            self._update_count += 1

    def score_action(self, action: str) -> float:
        return float(self._action_scores.get(str(action).strip(), 0.0))

    def read_stats(self) -> GTTAStats:
        return GTTAStats(
            queued_trajectories=len(self._queue),
            dropped_trajectories=int(self._dropped_trajectories),
            processed_trajectories=int(self._processed_trajectories),
            update_count=int(self._update_count),
        )
