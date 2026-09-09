"""Deterministic instance-owned GTTA dynamics memory."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import json

from pllm.workloads.gtta.types import (
    DynamicsEvent,
    GTTAConfig,
    GTTAHostTransaction,
    GTTAQuery,
    GTTAQueryResult,
    GTTAResult,
    GTTATrajectoryBatch,
    GTTAUpdateReceipt,
    PromptContext,
    canonical_state_signature,
)


@dataclass(frozen=True, slots=True)
class GTTADynamicsSnapshot:
    state_action_scores: dict[tuple[str, str, str], float]
    records: tuple[DynamicsEvent, ...]
    version: int


class GTTADynamicsState:
    """Sequential EMA scores and immutable event history for one instance."""

    __slots__ = ("_records", "_state_action_scores", "_version")

    def __init__(self) -> None:
        self._state_action_scores: dict[tuple[str, str, str], float] = {}
        self._records: list[DynamicsEvent] = []
        self._version = 0

    @property
    def version(self) -> int:
        return self._version

    def prepare(self, transaction: GTTAHostTransaction) -> GTTAStepContext:
        return GTTAStepContext(transaction, self)

    def output(self, context: GTTAStepContext) -> GTTAResult:
        if context.result is None:
            raise RuntimeError("GTTA signed graph did not produce its outputs")
        return context.result

    def snapshot(self) -> GTTADynamicsSnapshot:
        return GTTADynamicsSnapshot(
            dict(self._state_action_scores),
            tuple(self._records),
            self._version,
        )

    def restore(self, snapshot: GTTADynamicsSnapshot) -> None:
        self._state_action_scores = dict(snapshot.state_action_scores)
        self._records = list(snapshot.records)
        self._version = snapshot.version

    def update(
        self,
        batch: GTTATrajectoryBatch,
        config: GTTAConfig,
    ) -> GTTAUpdateReceipt:
        for event in batch.entries:
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
        # Stable sorting retains candidate/history order for equal keys without ordinal copies.
        ranked = tuple(sorted(((action, self._state_action_scores.get((query.task_id, signature, action), 0.0))
            for action in query.candidate_actions), key=lambda item: -item[1]))
        candidate_set = frozenset(query.candidate_actions)
        ordered = sorted((
            record
            for record in self._records
            if record.task_id == query.task_id
            and (not query.action or record.action == query.action)
            and (not signature or record.state_signature == signature)
            and (not candidate_set or record.action in candidate_set)
        ), key=lambda record: (-record.reward, record.action, record.state_signature, record.next_observation))
        records = tuple(islice(ordered, query.max_records or None))
        del ordered
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


@dataclass(slots=True)
class GTTAStepContext:
    transaction: GTTAHostTransaction
    dynamics: GTTADynamicsState
    update: GTTAUpdateReceipt | None = None
    queried: GTTAQueryResult | None = None
    result: GTTAResult | None = None


def update(config: GTTAConfig, context: GTTAStepContext) -> None:
    context.update = context.dynamics.update(context.transaction.trajectories, config)


def query(context: GTTAStepContext) -> None:
    context.queried = context.dynamics.query(context.transaction.query)


def publish(context: GTTAStepContext) -> None:
    if context.update is None or context.queried is None:
        raise RuntimeError("GTTA publish requires update and query outputs")
    context.result = GTTAResult(context.transaction.transaction_id, context.update, context.queried)
