"""Native GTTA trajectory-memory workload.

Environment execution is an explicit typed host service.  State mutation and
querying are signed extern tasks whose ordering follows state effects; the
workload contains no placement or scheduling policy.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
import torch

from tllm.kit.workload import Workload
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.trajectory_store import PromptContextEntry, TrajectoryEvent
from tllm.workload import (
    Before,
    ConfigField,
    ConfigKind,
    HostOutput,
    NumericConstraint,
    ServiceState,
    State,
    StagedPort,
    WorkloadConfig,
    config_field,
    declared_config_field,
    extern,
    host_output,
    maybe_transition,
    read,
    staged,
    task,
    workload,
)


@dataclass(frozen=True, slots=True)
class TrajectoryFeedback:
    task_id: str
    action: str
    observation: str = ""
    reward: float = 0.0
    previous_observation: str | None = None
    next_observation: str = ""
    environmental_dynamics: str = ""

    @property
    def previous_observation_text(self) -> str:
        return (
            self.observation
            if self.previous_observation is None
            else self.previous_observation
        )


@dataclass(frozen=True, slots=True)
class DynamicsEvent:
    task_id: str
    state_signature: str
    action: str
    previous_observation: str
    next_observation: str
    environmental_dynamics: str
    reward: float


@dataclass(frozen=True, slots=True)
class GTTATrajectoryBatch:
    entries: tuple[DynamicsEvent, ...]


@dataclass(frozen=True, slots=True)
class GTTAQuery:
    prompt_idx: int
    task_id: str
    candidate_actions: tuple[str, ...]
    observation: str = ""
    state_signature: str = ""
    action: str = ""
    max_records: int = 0
    position: str = "prefix"


@dataclass(frozen=True, slots=True)
class GTTAUpdateReceipt:
    processed: int
    state_version: int


@dataclass(frozen=True, slots=True)
class GTTAQueryResult:
    ranked_actions: tuple[tuple[str, float], ...]
    records: tuple[DynamicsEvent, ...]
    prompt_context: PromptContextEntry


_GTTA_ENABLED = ConfigField.boolean(True)


class GTTAConfig(WorkloadConfig):
    enabled: bool = declared_config_field(_GTTA_ENABLED)
    max_queue_size: int = config_field(
        default=128, kind=ConfigKind.INT, constraint=NumericConstraint.NONNEGATIVE
    )
    score_momentum: float = config_field(
        default=0.0,
        kind=ConfigKind.FLOAT,
        constraint=NumericConstraint.UNIT_INTERVAL,
    )


class GTTAEnvironmentService:
    """Bounded typed queue between an environment and the signed workload."""

    def __init__(self, config: GTTAConfig) -> None:
        self.config = config
        self._queue: deque[DynamicsEvent] = deque()
        self.dropped_trajectories = 0

    def enqueue_feedback(self, feedback: TrajectoryFeedback) -> bool:
        previous = feedback.previous_observation_text
        return self._enqueue(
            DynamicsEvent(
                task_id=feedback.task_id.strip(),
                state_signature=GTTADynamicsState.state_signature(previous),
                action=feedback.action.strip(),
                previous_observation=previous,
                next_observation=feedback.next_observation,
                environmental_dynamics=feedback.environmental_dynamics,
                reward=feedback.reward,
            )
        )

    def enqueue_event(self, event: TrajectoryEvent) -> bool:
        return self._enqueue(
            DynamicsEvent(
                task_id=event.task_id,
                state_signature=event.state_signature,
                action=event.action,
                previous_observation=event.previous_observation,
                next_observation=event.next_observation,
                environmental_dynamics="",
                reward=0.0 if event.reward is None else event.reward,
            )
        )

    def _enqueue(self, event: DynamicsEvent) -> bool:
        if len(self._queue) == self.config.max_queue_size:
            self.dropped_trajectories += 1
            return False
        self._queue.append(event)
        return True

    def drain(self, *, limit: int = 0) -> GTTATrajectoryBatch:
        count = len(self._queue) if limit <= 0 else min(limit, len(self._queue))
        return GTTATrajectoryBatch(
            tuple(self._queue.popleft() for _ in range(count))
        )


class GTTADynamicsState(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_scores: dict[str, float] = defaultdict(float)
        self.state_action_scores: dict[tuple[str, str, str], float] = defaultdict(float)
        self.records: list[DynamicsEvent] = []

    @staticmethod
    def state_signature(observation: str) -> str:
        return " ".join(observation.split()).casefold()

    def update(self, batch: GTTATrajectoryBatch, momentum: float) -> int:
        for event in batch.entries:
            self.action_scores[event.action] = (
                self.action_scores[event.action] * momentum
                + event.reward * (1.0 - momentum)
            )
            state_key = (event.task_id, event.state_signature, event.action)
            self.state_action_scores[state_key] = (
                self.state_action_scores[state_key] * momentum
                + event.reward * (1.0 - momentum)
            )
            self.records.append(event)
        return len(batch.entries)

    def query(self, query: GTTAQuery) -> GTTAQueryResult:
        signature = query.state_signature or self.state_signature(query.observation)
        scored = tuple(
            (
                action,
                self.state_action_scores.get((query.task_id, signature, action), 0.0),
                ordinal,
            )
            for ordinal, action in enumerate(query.candidate_actions)
        )
        ranked = tuple(
            (action, score)
            for action, score, _ordinal in sorted(
                scored, key=lambda item: (-item[1], item[2])
            )
        )
        candidate_set = frozenset(query.candidate_actions)
        matching = tuple(
            record
            for record in self.records
            if record.task_id == query.task_id
            and (not query.action or record.action == query.action)
            and (not signature or record.state_signature == signature)
            and (not candidate_set or record.action in candidate_set)
        )
        records = tuple(
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
            )[: query.max_records or None]
        )
        context = "\n".join(
            json.dumps(
                {
                    "action_taken": record.action,
                    "environmental_dynamics": (
                        record.environmental_dynamics
                        or f"Next observation: {record.next_observation}"
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
            ranked_actions=ranked,
            records=records,
            prompt_context=PromptContextEntry(
                prompt_idx=query.prompt_idx,
                text=context,
                position=query.position,
            ),
        )


class GTTAWorkload(Workload[GTTAConfig]):
    dynamics: ServiceState[GTTADynamicsState] = State[GTTADynamicsState].service(
        transition="trajectory_update"
    )
    trajectories: StagedPort[GTTATrajectoryBatch] = StagedPort()
    queries: StagedPort[GTTAQuery] = StagedPort()
    updates: HostOutput[GTTAUpdateReceipt] = HostOutput()
    query_results: HostOutput[GTTAQueryResult] = HostOutput()

    @dynamics.initialize
    def init_dynamics(self, _spec: RuntimeSpecializationSpec) -> GTTADynamicsState:
        return GTTADynamicsState()

    @task(
        inputs={"batch": staged(trajectories)},
        outputs={"receipt": host_output(updates)},
        effects=(maybe_transition(dynamics),),
        timing=Before("step.end"),
        implementation=extern("tllm.gtta:update_dynamics@1"),
    )
    def update(self, batch: GTTATrajectoryBatch) -> GTTAUpdateReceipt:
        processed = self.dynamics.update(batch, self.config.score_momentum)
        version = self.state_version
        if processed:
            version = self.advance_state(GTTAWorkload.dynamics)
        return GTTAUpdateReceipt(processed=processed, state_version=version)

    @task(
        inputs={"query": staged(queries)},
        outputs={"result": host_output(query_results)},
        effects=(read(dynamics),),
        timing=Before("next.prepare"),
        implementation=extern("tllm.gtta:query_dynamics@1"),
    )
    def query(self, query: GTTAQuery) -> GTTAQueryResult:
        return self.dynamics.query(query)


workload(GTTAWorkload, id="gtta", config=GTTAConfig, enabled=_GTTA_ENABLED)


__all__ = [
    "DynamicsEvent",
    "GTTAConfig",
    "GTTADynamicsState",
    "GTTAEnvironmentService",
    "GTTAQuery",
    "GTTAQueryResult",
    "GTTATrajectoryBatch",
    "GTTAUpdateReceipt",
    "GTTAWorkload",
    "TrajectoryFeedback",
]
