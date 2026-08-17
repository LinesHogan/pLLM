#!/usr/bin/env python3
"""Configuration and feedback contracts for the GTTA-inspired consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


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
    observation: str = ""
    reward: float | int | str | Mapping[str, Any] = 0.0
    previous_observation: str | None = None
    next_observation: str = ""
    environmental_dynamics: str = ""

    def __post_init__(self) -> None:
        if not str(self.task_id).strip():
            raise ValueError("TrajectoryFeedback requires task_id")
        if not str(self.action).strip():
            raise ValueError("TrajectoryFeedback requires action")
        parse_reward(self.reward)

    @property
    def reward_value(self) -> float:
        return parse_reward(self.reward)

    @property
    def previous_observation_text(self) -> str:
        if self.previous_observation is not None:
            return str(self.previous_observation)
        return str(self.observation)


@dataclass
class GTTAConsumerConfig:
    consumer_id: str = "gtta"
    enabled: bool = True
    max_queue_size: int = 128
    score_momentum: float = 0.0
