#!/usr/bin/env python3
"""GTTA-inspired trajectory feedback consumer."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tllm.consumers.base import BaseConsumer
from tllm.ports.base import ConsumerFlow
from tllm.runtime.trajectory_store import PromptContextEntry, TrajectoryEvent


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
        self._queue: deque[TrajectoryFeedback | TrajectoryEvent] = deque()
        self._dropped_trajectories = 0
        self._processed_trajectories = 0
        self._update_count = 0
        self._action_scores: dict[str, float] = defaultdict(float)
        self._action_counts: dict[str, int] = defaultdict(int)
        self._state_action_scores: dict[tuple[str, str, str], float] = defaultdict(float)
        self._state_action_counts: dict[tuple[str, str, str], int] = defaultdict(int)
        self._dynamics_records: list[dict[str, Any]] = []

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

    def enqueue_trajectory_event(self, event: TrajectoryEvent) -> bool:
        if not isinstance(event, TrajectoryEvent):
            raise TypeError("event must be TrajectoryEvent")
        if len(self._queue) >= max(0, int(self.config.max_queue_size)):
            self._dropped_trajectories += 1
            return False
        self._queue.append(event)
        return True

    def process_queued_trajectories(self, *, limit: int = 0) -> None:
        max_items = len(self._queue) if int(limit) <= 0 else min(int(limit), len(self._queue))
        for _ in range(max_items):
            item = self._queue.popleft()
            if not bool(self.config.enabled):
                continue
            if isinstance(item, TrajectoryEvent):
                self._process_transition(
                    task_id=item.task_id,
                    state_signature=item.state_signature,
                    action=item.action,
                    previous_observation=item.previous_observation,
                    next_observation=item.next_observation,
                    reward=0.0 if item.reward is None else item.reward,
                    environmental_dynamics="",
                )
                continue

            feedback = item
            task_id = str(feedback.task_id).strip()
            previous_observation = feedback.previous_observation_text
            self._process_transition(
                task_id=task_id,
                state_signature=self.state_signature(previous_observation),
                action=str(feedback.action).strip(),
                previous_observation=previous_observation,
                next_observation=str(feedback.next_observation),
                reward=float(feedback.reward_value),
                environmental_dynamics=str(feedback.environmental_dynamics),
            )

    def _process_transition(
        self,
        *,
        task_id: str,
        state_signature: str,
        action: str,
        previous_observation: str,
        next_observation: str,
        reward: float,
        environmental_dynamics: str,
    ) -> None:
        state_key = (task_id, state_signature, action)
        old = float(self._action_scores[action])
        momentum = min(1.0, max(0.0, float(self.config.score_momentum)))
        new_score = old * momentum + reward * (1.0 - momentum)
        self._action_scores[action] = new_score
        self._action_counts[action] += 1
        old_state_score = float(self._state_action_scores[state_key])
        new_state_score = old_state_score * momentum + reward * (1.0 - momentum)
        self._state_action_scores[state_key] = new_state_score
        self._state_action_counts[state_key] += 1
        self._dynamics_records.append(
            {
                "task_id": task_id,
                "state_signature": state_signature,
                "action": action,
                "previous_observation": previous_observation,
                "next_observation": next_observation,
                "environmental_dynamics": environmental_dynamics,
                "reward": reward,
            }
        )
        self._processed_trajectories += 1
        self._update_count += 1

    @staticmethod
    def state_signature(observation: str) -> str:
        return " ".join(str(observation).split()).casefold()

    def score_action(
        self,
        action: str,
        *,
        task_id: str | None = None,
        observation: str | None = None,
        state_signature: str | None = None,
    ) -> float:
        normalized_action = str(action).strip()
        if task_id is not None and (observation is not None or state_signature is not None):
            signature = state_signature if state_signature is not None else self.state_signature(str(observation))
            key = (str(task_id).strip(), str(signature), normalized_action)
            return float(self._state_action_scores.get(key, 0.0))
        return float(self._action_scores.get(str(action).strip(), 0.0))

    def rank_candidate_actions(
        self,
        *,
        task_id: str,
        candidate_actions: Sequence[str],
        observation: str | None = None,
        state_signature: str | None = None,
    ) -> list[tuple[str, float]]:
        signature = state_signature if state_signature is not None else self.state_signature(str(observation or ""))
        scored = [
            (
                str(action),
                self.score_action(str(action), task_id=task_id, state_signature=signature),
                index,
            )
            for index, action in enumerate(candidate_actions)
        ]
        scored.sort(key=lambda item: (-item[1], item[2]))
        return [(action, score) for action, score, _index in scored]

    def export_dynamics_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._dynamics_records]

    def query_dynamics_records(
        self,
        *,
        task_id: str | None = None,
        action: str | None = None,
        observation: str | None = None,
        state_signature: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_task_id = None if task_id is None else str(task_id).strip()
        normalized_action = None if action is None else str(action).strip()
        normalized_signature = state_signature
        if normalized_signature is None and observation is not None:
            normalized_signature = self.state_signature(str(observation))

        records: list[dict[str, Any]] = []
        for record in self._dynamics_records:
            if normalized_task_id is not None and record["task_id"] != normalized_task_id:
                continue
            if normalized_action is not None and record["action"] != normalized_action:
                continue
            if normalized_signature is not None and record["state_signature"] != str(normalized_signature):
                continue
            records.append(dict(record))
        return records

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
        records = self._rank_dynamics_records(
            task_id=task_id,
            action=action,
            observation=observation,
            state_signature=state_signature,
            candidate_actions=candidate_actions,
            max_records=max_records,
        )
        return tuple(self._format_dynamics_snippet(record) for record in records)

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
        records = self._rank_dynamics_records(
            task_id=task_id,
            action=action,
            observation=observation,
            state_signature=state_signature,
            candidate_actions=candidate_actions,
            max_records=max_records,
        )
        return "\n".join(json.dumps(self._to_environment_dynamics_payload(record), sort_keys=True) for record in records)

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
        snippets = self.render_ranked_dynamics_snippets(
            task_id=task_id,
            action=action,
            observation=observation,
            state_signature=state_signature,
            candidate_actions=candidate_actions,
            max_records=max_records,
        )
        return PromptContextEntry(prompt_idx=prompt_idx, text="\n".join(snippets), position=position)

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
        text = self.render_environment_dynamics_context(
            task_id=task_id,
            action=action,
            observation=observation,
            state_signature=state_signature,
            candidate_actions=candidate_actions,
            max_records=max_records,
        )
        return PromptContextEntry(prompt_idx=prompt_idx, text=text, position=position)

    def _rank_dynamics_records(
        self,
        *,
        task_id: str,
        action: str | None = None,
        observation: str | None = None,
        state_signature: str | None = None,
        candidate_actions: Sequence[str] | None = None,
        max_records: int = 0,
    ) -> list[dict[str, Any]]:
        candidate_set = None if candidate_actions is None else {str(candidate).strip() for candidate in candidate_actions}
        records = self.query_dynamics_records(
            task_id=task_id,
            action=action,
            observation=observation,
            state_signature=state_signature,
        )
        if candidate_set is not None:
            records = [record for record in records if record["action"] in candidate_set]
        ranked = sorted(
            enumerate(records),
            key=lambda item: (
                -float(item[1]["reward"]),
                str(item[1]["action"]),
                str(item[1]["state_signature"]),
                str(item[1]["next_observation"]),
                item[0],
            ),
        )
        limit = int(max_records)
        if limit < 0:
            raise ValueError("max_records must be non-negative")
        if limit > 0:
            ranked = ranked[:limit]
        return [record for _index, record in ranked]

    @staticmethod
    def _format_dynamics_snippet(record: Mapping[str, Any]) -> str:
        return (
            f"task={record['task_id']} state={record['state_signature']} action={record['action']} "
            f"reward={float(record['reward']):.6g} next={record['next_observation']}"
        )

    @staticmethod
    def _to_environment_dynamics_payload(record: Mapping[str, Any]) -> dict[str, object]:
        environmental_dynamics = str(record["environmental_dynamics"])
        if not environmental_dynamics:
            environmental_dynamics = f"Next observation: {record['next_observation']}"
        return {
            "initial_state": str(record["previous_observation"]),
            "action_taken": str(record["action"]),
            "environmental_dynamics": environmental_dynamics,
            "task_id": str(record["task_id"]),
            "state_signature": str(record["state_signature"]),
            "reward": float(record["reward"]),
        }

    def score_global_action(self, action: str) -> float:
        return float(self._action_scores.get(str(action).strip(), 0.0))

    def read_stats(self) -> GTTAStats:
        return GTTAStats(
            queued_trajectories=len(self._queue),
            dropped_trajectories=int(self._dropped_trajectories),
            processed_trajectories=int(self._processed_trajectories),
            update_count=int(self._update_count),
        )
