"""Deterministic instance-owned GTTA dynamics memory."""

from __future__ import annotations

from dataclasses import dataclass
import json

from tllm.canonical.workloads.gtta.types import (
    DynamicsEvent,
    GTTAConfig,
    GTTAQuery,
    GTTAQueryResult,
    GTTATrajectoryBatch,
    GTTAUpdateReceipt,
    PromptContext,
    canonical_state_signature,
)


@dataclass(frozen=True, slots=True)
class GTTADynamicsSnapshot:
    action_scores: dict[str, float]
    state_action_scores: dict[tuple[str, str, str], float]
    records: tuple[DynamicsEvent, ...]
    version: int


class GTTADynamicsState:
    """Sequential EMA scores and immutable event history for one instance."""

    __slots__ = ("_action_scores", "_records", "_state_action_scores", "_version")

    def __init__(self) -> None:
        self._action_scores: dict[str, float] = {}
        self._state_action_scores: dict[tuple[str, str, str], float] = {}
        self._records: list[DynamicsEvent] = []
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def snapshot(self) -> GTTADynamicsSnapshot:
        return GTTADynamicsSnapshot(
            dict(self._action_scores),
            dict(self._state_action_scores),
            tuple(self._records),
            self._version,
        )

    def restore(self, snapshot: GTTADynamicsSnapshot) -> None:
        self._action_scores = dict(snapshot.action_scores)
        self._state_action_scores = dict(snapshot.state_action_scores)
        self._records = list(snapshot.records)
        self._version = snapshot.version

    def update(
        self,
        batch: GTTATrajectoryBatch,
        config: GTTAConfig,
    ) -> GTTAUpdateReceipt:
        for event in batch.entries:
            self._action_scores[event.action] = (
                self._action_scores.get(event.action, 0.0) * config.score_momentum
                + event.reward * (1.0 - config.score_momentum)
            )
            state_key = (event.task_id, event.state_signature, event.action)
            self._state_action_scores[state_key] = (
                self._state_action_scores.get(state_key, 0.0)
                * config.score_momentum
                + event.reward * (1.0 - config.score_momentum)
            )
            self._records.append(event)
        if batch.entries:
            self._version += 1
        return GTTAUpdateReceipt(len(batch.entries), self._version)

    def query(self, query: GTTAQuery) -> GTTAQueryResult:
        signature = query.state_signature or canonical_state_signature(
            query.observation
        )
        scored = tuple(
            (
                action,
                self._state_action_scores.get(
                    (query.task_id, signature, action),
                    0.0,
                ),
                ordinal,
            )
            for ordinal, action in enumerate(query.candidate_actions)
        )
        ranked = tuple(
            (action, score)
            for action, score, _ordinal in sorted(
                scored,
                key=lambda item: (-item[1], item[2]),
            )
        )
        candidate_set = frozenset(query.candidate_actions)
        matching = tuple(
            record
            for record in self._records
            if record.task_id == query.task_id
            and (not query.action or record.action == query.action)
            and (not signature or record.state_signature == signature)
            and (not candidate_set or record.action in candidate_set)
        )
        ordered = tuple(
            record
            for _ordinal, record in sorted(
                enumerate(matching),
                key=lambda item: (
                    -item[1].reward,
                    item[1].action,
                    item[1].state_signature,
                    item[1].next_observation,
                    item[0],
                ),
            )
        )
        records = (
            ordered
            if query.max_records == 0
            else ordered[: query.max_records]
        )
        context = "\n".join(
            json.dumps(
                {
                    "action_taken": record.action,
                    "environmental_dynamics": (
                        record.environmental_dynamics
                        or "Next observation: " + record.next_observation
                    ),
                    "initial_state": record.previous_observation,
                    "reward": record.reward,
                    "state_signature": record.state_signature,
                    "task_id": record.task_id,
                },
                sort_keys=True,
            )
            for record in records
        )
        return GTTAQueryResult(
            ranked,
            records,
            PromptContext(query.prompt_idx, context, query.position),
        )


__all__ = [
    "GTTADynamicsSnapshot",
    "GTTADynamicsState",
]
