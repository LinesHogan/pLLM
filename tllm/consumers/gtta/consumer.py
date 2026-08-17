#!/usr/bin/env python3
"""BaseConsumer wrapper around the GTTA trajectory feedback engine."""

from __future__ import annotations

from typing import Sequence

from tllm.consumers.base import BaseConsumer
from tllm.consumers.gtta.config import GTTAConsumerConfig, TrajectoryFeedback, parse_reward
from tllm.consumers.gtta.engine import DynamicsEvent, GTTAEngine, GTTAStats
from tllm.ports.base import ConsumerFlow
from tllm.runtime.trajectory_store import PromptContextEntry, TrajectoryEvent

__all__ = [
    "GTTAConsumer",
    "GTTAConsumerConfig",
    "GTTAEngine",
    "GTTAStats",
    "TrajectoryFeedback",
    "parse_reward",
]


class GTTAConsumer(BaseConsumer):
    """Local action-score updater for environment trajectory feedback."""

    def __init__(self, config: GTTAConsumerConfig, *, engine: GTTAEngine | None = None) -> None:
        self.config = config
        self.engine = engine or GTTAEngine(config)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        return ()

    def enqueue_trajectory(self, feedback: TrajectoryFeedback) -> bool:
        return self.engine.enqueue_trajectory(feedback)

    def enqueue_trajectory_event(self, event: TrajectoryEvent) -> bool:
        return self.engine.enqueue_trajectory_event(event)

    def process_queued_trajectories(self, *, limit: int = 0) -> None:
        self.engine.process_queued_trajectories(limit=limit)

    @staticmethod
    def state_signature(observation: str) -> str:
        return GTTAEngine.state_signature(observation)

    def score_action(
        self,
        action: str,
        *,
        task_id: str | None = None,
        observation: str | None = None,
        state_signature: str | None = None,
    ) -> float:
        return self.engine.score_action(
            action,
            task_id=task_id,
            observation=observation,
            state_signature=state_signature,
        )

    def rank_candidate_actions(
        self,
        *,
        task_id: str,
        candidate_actions: Sequence[str],
        observation: str | None = None,
        state_signature: str | None = None,
    ) -> list[tuple[str, float]]:
        return self.engine.rank_candidate_actions(
            task_id=task_id,
            candidate_actions=candidate_actions,
            observation=observation,
            state_signature=state_signature,
        )

    def export_dynamics_records(self) -> tuple[DynamicsEvent, ...]:
        return self.engine.export_dynamics_records()

    def query_dynamics_records(
        self,
        *,
        task_id: str | None = None,
        action: str | None = None,
        observation: str | None = None,
        state_signature: str | None = None,
    ) -> tuple[DynamicsEvent, ...]:
        return self.engine.query_dynamics_records(
            task_id=task_id,
            action=action,
            observation=observation,
            state_signature=state_signature,
        )

    def render_ranked_dynamics_snippets(
        self,
        *,
        task_id: str,
        action: str | None = None,
        observation: str | None = None,
        state_signature: str | None = None,
        candidate_actions: Sequence[str] | None = None,
        max_records: int = 0,
    ) -> tuple[str, ...]:
        return self.engine.render_ranked_dynamics_snippets(
            task_id=task_id,
            action=action,
            observation=observation,
            state_signature=state_signature,
            candidate_actions=candidate_actions,
            max_records=max_records,
        )

    def render_environment_dynamics_context(
        self,
        *,
        task_id: str,
        action: str | None = None,
        observation: str | None = None,
        state_signature: str | None = None,
        candidate_actions: Sequence[str] | None = None,
        max_records: int = 0,
    ) -> str:
        return self.engine.render_environment_dynamics_context(
            task_id=task_id,
            action=action,
            observation=observation,
            state_signature=state_signature,
            candidate_actions=candidate_actions,
            max_records=max_records,
        )

    def build_prompt_context_entry(
        self,
        *,
        prompt_idx: int,
        task_id: str,
        action: str | None = None,
        observation: str | None = None,
        state_signature: str | None = None,
        candidate_actions: Sequence[str] | None = None,
        max_records: int = 0,
        position: str = "prefix",
    ) -> PromptContextEntry:
        return self.engine.build_prompt_context_entry(
            prompt_idx=prompt_idx,
            task_id=task_id,
            action=action,
            observation=observation,
            state_signature=state_signature,
            candidate_actions=candidate_actions,
            max_records=max_records,
            position=position,
        )

    def build_environment_dynamics_prompt_context_entry(
        self,
        *,
        prompt_idx: int,
        task_id: str,
        action: str | None = None,
        observation: str | None = None,
        state_signature: str | None = None,
        candidate_actions: Sequence[str] | None = None,
        max_records: int = 0,
        position: str = "prefix",
    ) -> PromptContextEntry:
        return self.engine.build_environment_dynamics_prompt_context_entry(
            prompt_idx=prompt_idx,
            task_id=task_id,
            action=action,
            observation=observation,
            state_signature=state_signature,
            candidate_actions=candidate_actions,
            max_records=max_records,
            position=position,
        )

    def score_global_action(self, action: str) -> float:
        return self.engine.score_global_action(action)

    def read_stats(self) -> GTTAStats:
        return self.engine.read_stats()
