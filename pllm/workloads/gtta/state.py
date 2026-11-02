"""GTTA's dynamics memory: EMA scores per (task, state, action) and the event history queries rank from."""

from __future__ import annotations

import json
from itertools import islice

from pllm.workloads.gtta.types import (
    DynamicsEvent,
    GTTAConfig,
    GTTAQuery,
    GTTAQueryResult,
    PromptContext,
    state_signature,
)


class GTTADynamicsState:
    """Sequential EMA scores and immutable event history for one instance."""

    __slots__ = ("_records", "_state_action_scores")

    def __init__(self) -> None:
        self._state_action_scores: dict[tuple[str, str, str], float] = {}
        self._records: list[DynamicsEvent] = []

    def update(
        self,
        events: tuple[DynamicsEvent, ...],
        config: GTTAConfig,
    ) -> None:
        """Fold events into the host dictionary in FIFO order."""
        for event in events:
            state_key = (event.task_id, event.state_signature, event.action)
            self._state_action_scores[state_key] = self._state_action_scores.get(
                state_key, 0.0
            ) * config.score_momentum + event.reward * (1.0 - config.score_momentum)
            self._records.append(event)

    def query(self, query: GTTAQuery) -> GTTAQueryResult:
        """Rank actions and render matching event history for a host query."""
        signature = query.state_signature or state_signature(query.observation)
        # Stable sorting retains candidate/history order for equal keys without ordinal copies.
        ranked = tuple(
            sorted(
                (
                    (action, self._state_action_scores.get((query.task_id, signature, action), 0.0))
                    for action in query.candidate_actions
                ),
                key=lambda item: -item[1],
            )
        )
        candidate_set = frozenset(query.candidate_actions)
        ordered = sorted(
            (
                record
                for record in self._records
                if record.task_id == query.task_id
                and (not query.action or record.action == query.action)
                and (not signature or record.state_signature == signature)
                and (not candidate_set or record.action in candidate_set)
            ),
            key=lambda record: (-record.reward, record.action, record.state_signature, record.next_observation),
        )
        records = tuple(islice(ordered, query.max_records or None))
        del ordered
        context = "\n".join(
            json.dumps(
                {
                    "action_taken": record.action,
                    "environmental_dynamics": (
                        record.environmental_dynamics or "Next observation: " + record.next_observation
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
            PromptContext(query.prompt_index, context, query.position),
        )
