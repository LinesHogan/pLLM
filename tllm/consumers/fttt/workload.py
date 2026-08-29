"""Native feedback-conditioned test-time training workload.

The host service judges completed trajectories and stages one atomic typed
batch. Tensor selection, feature construction, loss, optimization, and edit
prediction are signed tasks. Placement and overlap remain scheduler choices.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import re
from typing import cast

import torch
from torch.nn import functional as F

from tllm.consumers._common.alignment import LowRankAdapter
from tllm.consumers.fttt.feedback import (
    FTTTInstanceState,
    FTTTJudge,
    FTTTJudgeResult,
    FeedbackRecord,
    _default_trace_judge,
    edit_inputs,
    is_positive_feedback,
    target_delta,
)
from tllm.kit.workload import Workload
from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeEntry
from tllm.runtime.retry_control_runtime import RetryControlBatch, RetryControlEntry
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch, UpdateBarrierEntry
from tllm.workload import (
    AdamSpec,
    Before,
    Between,
    ConfigField,
    ConfigKind,
    HostOutput,
    NumericConstraint,
    State,
    StagedPort,
    TrainableState,
    WorkloadConfig,
    config_field,
    extern,
    extern_io,
    host_output,
    maybe_optimize,
    read,
    result,
    staged,
    task,
    temporary,
    workload,
)


_FTTT_ENABLED = ConfigField.boolean(True)
_FTTT_LEARNING_RATE = ConfigField.floating(1e-3, NumericConstraint.POSITIVE)


class FTTTConfig(WorkloadConfig):
    enabled: bool = cast(bool, _FTTT_ENABLED)
    hidden_size: int = config_field(
        default=0, kind=ConfigKind.INT, constraint=NumericConstraint.NONNEGATIVE
    )
    rank: int = config_field(
        default=16, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE
    )
    learning_rate: float = cast(float, _FTTT_LEARNING_RATE)
    max_queue_size: int = config_field(
        default=128, kind=ConfigKind.INT, constraint=NumericConstraint.NONNEGATIVE
    )
    feedback_feature_size: int = config_field(
        default=32, kind=ConfigKind.INT, constraint=NumericConstraint.POSITIVE
    )
    positive_reward_threshold: float = config_field(default=1.0, kind=ConfigKind.FLOAT)
    max_retries: int = config_field(
        default=0, kind=ConfigKind.INT, constraint=NumericConstraint.NONNEGATIVE
    )
    edit_scale: float = config_field(
        default=1.0,
        kind=ConfigKind.FLOAT,
        constraint=NumericConstraint.NONNEGATIVE,
    )


@dataclass(frozen=True, slots=True)
class FTTTLifecycleTransition:
    prompt_idx: int
    request_id: str
    previous: FTTTInstanceState
    next: FTTTInstanceState
    trained: bool


@dataclass(frozen=True, slots=True)
class FTTTClosedLoopIntent:
    transaction_id: int
    reward_outcomes: RewardOutcomeBatch
    retry_controls: RetryControlBatch
    update_barriers: UpdateBarrierBatch
    feedback_records: tuple[FeedbackRecord, ...]
    transitions: tuple[FTTTLifecycleTransition, ...]
    adapter_version: int = 0


@dataclass(frozen=True, slots=True)
class FTTTTrainingPlan:
    selected_rows: tuple[int, ...]
    intent: FTTTClosedLoopIntent


@dataclass(frozen=True, slots=True)
class FTTTTrainingBatch:
    hidden_rows: torch.Tensor
    plan: FTTTTrainingPlan


@dataclass(frozen=True, slots=True)
class _InstanceKey:
    prompt_idx: int
    request_id: str


@dataclass(frozen=True, slots=True)
class _LifecycleDecision:
    adaptable: bool
    next_state: FTTTInstanceState
    action: str
    reason: str
    retry_after_update: bool
    barrier_state: str
    barrier_reason: str
    ready_count: int


class FTTTFeedbackService:
    """Transactional judge/queue service for atomic native batches."""

    def __init__(self, config: FTTTConfig, *, judge: FTTTJudge | None = None) -> None:
        self.config = config
        self.judge: FTTTJudge = _default_trace_judge if judge is None else judge
        self._instances: dict[_InstanceKey, FTTTInstanceState] = {}
        self._projected_instances: dict[_InstanceKey, FTTTInstanceState] = {}
        self._queue: deque[FTTTTrainingBatch] = deque()
        self._active: FTTTTrainingBatch | None = None
        self._next_transaction_id = 1
        self._queued_feedback = 0
        self.dropped_feedback = 0

    @property
    def queued_feedback(self) -> int:
        return self._queued_feedback

    def instance_state(self, prompt_idx: int, *, request_id: str = "") -> FTTTInstanceState:
        if request_id:
            return self._instances.get(
                _InstanceKey(prompt_idx, request_id), FTTTInstanceState()
            )
        states = tuple(
            state
            for key, state in self._instances.items()
            if key.prompt_idx == prompt_idx
        )
        if not states:
            return FTTTInstanceState()
        return FTTTInstanceState(
            retries=max(state.retries for state in states),
            stopped=any(state.stopped for state in states),
            adapter_version=max(state.adapter_version for state in states),
        )

    def _decision(
        self, record: FeedbackRecord, current: FTTTInstanceState
    ) -> _LifecycleDecision:
        if current.stopped:
            return _LifecycleDecision(
                False, current, "stop", "instance-stopped", False,
                "blocked", "instance-stopped", 0,
            )
        if is_positive_feedback(
            record,
            positive_reward_threshold=self.config.positive_reward_threshold,
        ):
            return _LifecycleDecision(
                False, replace(current, stopped=True), "accept",
                "positive-feedback", False, "blocked", "positive-feedback", 0,
            )
        retries = current.retries + 1
        stopped = self.config.max_retries > 0 and retries >= self.config.max_retries
        return _LifecycleDecision(
            True,
            FTTTInstanceState(retries, stopped, current.adapter_version),
            "stop" if stopped else "retry",
            "retry-limit" if stopped else "feedback-update",
            not stopped,
            "ready",
            "feedback-update-ready",
            1,
        )

    @staticmethod
    def _feedback_token_ids(text: str) -> tuple[int, ...]:
        tokens = re.findall(r"[a-z0-9_]+", text.lower()) or [text]
        return tuple(
            int.from_bytes(
                hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(),
                byteorder="little",
            )
            for token in tokens
        )

    def _record(
        self,
        entry: CompletionTraceEntry,
        outcome: RewardOutcomeEntry,
        judged: FTTTJudgeResult,
    ) -> FeedbackRecord:
        return FeedbackRecord(
            prompt_idx=entry.prompt_idx,
            feedback=outcome.feedback_text,
            reward=outcome.reward if outcome.reward is not None else 0.0,
            request_id=entry.request_id,
            sample_idx=entry.sample_idx,
            output_index=entry.output_index,
            group_id=entry.group_id,
            failed_text=entry.text,
            failed_token_ids=entry.token_ids,
            feedback_token_ids=(
                judged.feedback_token_ids
                or self._feedback_token_ids(outcome.feedback_text)
            ),
        )

    def enqueue(self, trace: GenerationTraceBatch, hidden_rows: torch.Tensor) -> bool:
        if hidden_rows.ndim != 2 or hidden_rows.shape[0] != len(trace.entries):
            raise ValueError("one hidden row is required for each completed trajectory")
        transaction_id = self._next_transaction_id
        self._next_transaction_id += 1
        outcomes: list[RewardOutcomeEntry] = []
        retries: list[RetryControlEntry] = []
        barriers: list[UpdateBarrierEntry] = []
        records: list[FeedbackRecord] = []
        transitions: list[FTTTLifecycleTransition] = []
        selected_rows: list[int] = []
        available = self.config.max_queue_size - self._queued_feedback
        for row, entry in enumerate(trace.entries):
            judged = self.judge(entry)
            feedback_text = judged.feedback_text or (
                "correct" if judged.is_correct else "Your answer is incorrect."
            )
            outcome = RewardOutcomeEntry(
                request_id=entry.request_id,
                prompt_idx=entry.prompt_idx,
                sample_idx=entry.sample_idx,
                output_index=entry.output_index,
                group_id=entry.group_id,
                reward=judged.reward,
                judge=judged.judge,
                label=judged.label,
                feedback_text=feedback_text,
                skip_reason=judged.skip_reason or (
                    "correct-output" if judged.is_correct else ""
                ),
                is_correct=judged.is_correct,
            )
            outcomes.append(outcome)
            key = _InstanceKey(entry.prompt_idx, entry.request_id)
            previous = self._projected_instances.get(
                key, self._instances.get(key, FTTTInstanceState())
            )
            if judged.is_correct:
                decision = _LifecycleDecision(
                    False, replace(previous, stopped=True), "accept", "correct-output",
                    False, "blocked", "correct-output", 0,
                )
                trained = False
            else:
                record = self._record(entry, outcome, judged)
                decision = self._decision(record, previous)
                trained = decision.adaptable and len(records) < available
                if trained:
                    records.append(record)
                    selected_rows.append(row)
                elif decision.adaptable:
                    self.dropped_feedback += 1
                    decision = _LifecycleDecision(
                        False, previous, "stop", "feedback-queue-full", False,
                        "blocked", "feedback-queue-full", 0,
                    )
            transition = FTTTLifecycleTransition(
                entry.prompt_idx,
                entry.request_id,
                previous,
                decision.next_state,
                trained,
            )
            transitions.append(transition)
            self._projected_instances[key] = decision.next_state
            retries.append(
                RetryControlEntry(
                    request_id=entry.request_id,
                    prompt_idx=entry.prompt_idx,
                    sample_idx=entry.sample_idx,
                    output_index=entry.output_index,
                    group_id=entry.group_id,
                    action=decision.action,
                    reason=decision.reason,
                    retry_after_update=decision.retry_after_update,
                )
            )
            barriers.append(
                UpdateBarrierEntry(
                    barrier_id=f"fttt:{entry.group_id}:{entry.prompt_idx}:{entry.sample_idx}",
                    group_id=entry.group_id,
                    consumer_id="fttt",
                    state=decision.barrier_state,
                    request_ids=(entry.request_id,),
                    ready_count=decision.ready_count,
                    reason=decision.barrier_reason,
                )
            )
        intent = FTTTClosedLoopIntent(
            transaction_id,
            RewardOutcomeBatch(tuple(outcomes)),
            RetryControlBatch(tuple(retries)),
            UpdateBarrierBatch(tuple(barriers)),
            tuple(records),
            tuple(transitions),
        )
        self._queue.append(
            FTTTTrainingBatch(
                hidden_rows,
                FTTTTrainingPlan(tuple(selected_rows), intent),
            )
        )
        self._queued_feedback += len(records)
        return bool(records)

    def next_batch(self) -> FTTTTrainingBatch | None:
        if self._active is not None:
            raise RuntimeError("the active FTTT transaction must be committed or aborted")
        if not self._queue:
            return None
        self._active = self._queue.popleft()
        self._queued_feedback -= len(self._active.plan.intent.feedback_records)
        return self._active

    def acknowledge(self, intent: FTTTClosedLoopIntent) -> None:
        active = self._active
        if active is None or active.plan.intent.transaction_id != intent.transaction_id:
            raise RuntimeError("FTTT acknowledgement does not match the active transaction")
        for transition in intent.transitions:
            key = _InstanceKey(transition.prompt_idx, transition.request_id)
            current = self._instances.get(key, FTTTInstanceState())
            if current != transition.previous:
                raise RuntimeError("FTTT lifecycle transition is stale")
            version = intent.adapter_version if transition.trained else current.adapter_version
            self._instances[key] = replace(transition.next, adapter_version=version)
        self._active = None

    def abort(self, transaction_id: int) -> None:
        active = self._active
        if active is None or active.plan.intent.transaction_id != transaction_id:
            raise RuntimeError("FTTT abort does not match the active transaction")
        self._active = None
        self._queue.clear()
        self._queued_feedback = 0
        self._projected_instances = dict(self._instances)


class FTTTWorkload(Workload[FTTTConfig]):
    edit_adapter: TrainableState[LowRankAdapter] = State[LowRankAdapter].replica(
        optimizer=AdamSpec.configured(_FTTT_LEARNING_RATE)
    )
    hidden_rows: StagedPort[torch.Tensor] = StagedPort()
    plans: StagedPort[FTTTTrainingPlan] = StagedPort()
    edit_delta: HostOutput[torch.Tensor] = HostOutput()
    reward_outcomes: HostOutput[RewardOutcomeBatch] = HostOutput()
    retry_controls: HostOutput[RetryControlBatch] = HostOutput()
    update_barriers: HostOutput[UpdateBarrierBatch] = HostOutput()
    closed_loop: HostOutput[FTTTClosedLoopIntent] = HostOutput()

    @edit_adapter.initialize
    def init_edit_adapter(self, spec: RuntimeSpecializationSpec) -> LowRankAdapter:
        hidden_size = self.config.hidden_size or spec.hidden_size
        return LowRankAdapter(
            in_dim=hidden_size + self.config.feedback_feature_size + 1,
            out_dim=hidden_size,
            rank=self.config.rank,
            device=spec.device,
            dtype=spec.dtype,
        )

    @task(
        inputs={"hidden_rows": staged(hidden_rows), "plan": staged(plans)},
        outputs={
            "edit_input": temporary(),
            "target": temporary(),
            "intent": temporary(),
        },
        timing=Between("token.commit", "step.end"),
        implementation=extern("tllm.fttt:prepare_edit_batch@1"),
    )
    def prepare(
        self, hidden_rows: torch.Tensor, plan: FTTTTrainingPlan
    ) -> tuple[torch.Tensor, torch.Tensor, FTTTClosedLoopIntent]:
        hidden_size = hidden_rows.shape[1]
        if plan.selected_rows:
            index = torch.tensor(
                plan.selected_rows,
                device=hidden_rows.device,
                dtype=torch.long,
            )
            hidden = hidden_rows.detach().index_select(0, index)
            edit_input = edit_inputs(
                hidden,
                plan.intent.feedback_records,
                feature_size=self.config.feedback_feature_size,
            )
            target = target_delta(
                hidden,
                plan.intent.feedback_records,
                feature_size=self.config.feedback_feature_size,
                edit_scale=self.config.edit_scale,
            )
        else:
            edit_input = hidden_rows.new_empty(
                (0, hidden_size + self.config.feedback_feature_size + 1)
            )
            target = hidden_rows.new_empty((0, hidden_size))
        return edit_input, target, plan.intent

    @task(
        inputs={
            "edit_input": result("prepare", "edit_input"),
            "target": result("prepare", "target"),
        },
        outputs={
            "loss": maybe_optimize(edit_adapter, at=Before("next.prepare")),
            "kept_input": temporary(),
        },
        timing=Between("token.commit", "step.end"),
    )
    def train(
        self, edit_input: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if edit_input.shape[0] == 0:
            return None, edit_input
        prediction = self.edit_adapter(edit_input)
        return F.mse_loss(prediction, target), edit_input

    @task(
        inputs={"edit_input": result("train", "kept_input")},
        outputs={"delta": host_output(edit_delta)},
        effects=(read(edit_adapter),),
        timing=Before("next.prepare"),
    )
    def predict(self, edit_input: torch.Tensor) -> torch.Tensor:
        return self.edit_adapter(edit_input)

    @task(
        inputs={"intent": result("prepare", "intent")},
        outputs={
            "reward_outcomes": host_output(reward_outcomes),
            "retry_controls": host_output(retry_controls),
            "update_barriers": host_output(update_barriers),
            "closed_loop": host_output(closed_loop),
        },
        effects=(read(edit_adapter), extern_io("fttt:closed-loop")),
        timing=Before("next.prepare"),
        implementation=extern("tllm.fttt:publish_closed_loop@1"),
    )
    def publish(
        self, intent: FTTTClosedLoopIntent
    ) -> tuple[
        RewardOutcomeBatch,
        RetryControlBatch,
        UpdateBarrierBatch,
        FTTTClosedLoopIntent,
    ]:
        versioned = replace(intent, adapter_version=self.state_version)
        return (
            versioned.reward_outcomes,
            versioned.retry_controls,
            versioned.update_barriers,
            versioned,
        )


workload(FTTTWorkload, id="fttt", config=FTTTConfig, enabled=_FTTT_ENABLED)


__all__ = [
    "FTTTClosedLoopIntent",
    "FTTTConfig",
    "FTTTFeedbackService",
    "FTTTLifecycleTransition",
    "FTTTTrainingBatch",
    "FTTTTrainingPlan",
    "FTTTWorkload",
]
