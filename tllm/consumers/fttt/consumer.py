#!/usr/bin/env python3
"""FTTT-inspired feedback edit consumer.

The judge, feedback queue, and per-(prompt, request) lifecycle are plain
Python; the tensor edit adapter is a compiled ``FTTTProgram`` driven with
caller-supplied edit inputs and target deltas.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Sequence

import torch
from torch.nn import functional as F

from tllm.consumers.base import BaseConsumer
from tllm.consumers._common.bundle_inputs import post_generate_input, provided_bundle
from tllm.consumers.fttt.config import FTTTConsumerConfig
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
from tllm.consumers.fttt.program import FTTTProgram
from tllm.ports.generation_trace import GenerationTrace
from tllm.ports.base import ConsumerFlow
from tllm.ports.reward_outcome import RewardOutcome
from tllm.ports.retry_control import RetryControl
from tllm.ports.update_barrier import UpdateBarrier
from tllm.program import ProgramConsumer, StateSpec, compile_program
from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeEntry
from tllm.runtime.retry_control_runtime import RetryControlBatch, RetryControlEntry
from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch, UpdateBarrierEntry
from tllm.workgraph.frontend import DynamicDim, StateScope


@dataclass(frozen=True)
class FTTTStats:
    queued_feedback: int = 0
    dropped_feedback: int = 0
    feedback_count: int = 0
    update_count: int = 0
    loss_count: int = 0
    last_loss: float = 0.0
    skipped_feedback: int = 0
    stopped_instances: int = 0
    adapter_version: int = 0


@dataclass(frozen=True)
class FTTTClosedLoopIntent:
    reward_outcomes: RewardOutcomeBatch
    retry_controls: RetryControlBatch
    update_barriers: UpdateBarrierBatch
    feedback_records: tuple[FeedbackRecord, ...]
    update_count: int


@dataclass(frozen=True)
class _FTTTInstanceKey:
    prompt_idx: int
    request_id: str = ""


@dataclass(frozen=True)
class _FTTTLifecycleDecision:
    adaptable: bool
    next_state: FTTTInstanceState
    retry_action: str
    retry_reason: str
    retry_after_update: bool
    barrier_state: str
    barrier_reason: str
    ready_count: int


class FTTTConsumer(BaseConsumer[FTTTConsumerConfig]):
    """Feedback-conditioned local edit consumer."""

    def __init__(
        self,
        config: FTTTConsumerConfig,
        *,
        judge: FTTTJudge | None = None,
    ) -> None:
        self.config = config
        self._judge = judge or _default_trace_judge
        self._queue: deque[FeedbackRecord] = deque()
        self._dropped_feedback = 0
        self._compiled: ProgramConsumer | None = None
        self._step_id = 0
        self._feedback_count = 0
        self._skipped_feedback = 0
        self._loss_sum_seen = 0.0
        self._last_loss = 0.0
        self._instances: dict[_FTTTInstanceKey, FTTTInstanceState] = {}

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    @property
    def compiled(self) -> ProgramConsumer | None:
        return self._compiled

    def flows(self) -> Sequence[ConsumerFlow]:
        return (
            ConsumerFlow(
                reads=(GenerationTrace.read(),),
                writes=(
                    RewardOutcome.write(),
                    RetryControl.write(),
                    UpdateBarrier.write(),
                ),
                window="out_of_band",
            ),
        )

    # -- feedback queue ------------------------------------------------------

    def enqueue_feedback(self, feedback: FeedbackRecord) -> bool:
        if len(self._queue) >= max(0, int(self.config.max_queue_size)):
            self._dropped_feedback += 1
            return False
        self._queue.append(feedback)
        return True

    def drain_feedback(self, *, limit: int = 0) -> list[FeedbackRecord]:
        max_items = len(self._queue) if int(limit) <= 0 else min(int(limit), len(self._queue))
        return [self._queue.popleft() for _ in range(max_items)]

    # -- per-instance lifecycle ----------------------------------------------

    def _instance_key(self, *, prompt_idx: int, request_id: str = "") -> _FTTTInstanceKey:
        request_id = str(request_id)
        if not request_id:
            return _FTTTInstanceKey(prompt_idx=int(prompt_idx))
        return _FTTTInstanceKey(prompt_idx=int(prompt_idx), request_id=request_id)

    def _instance_key_for_record(self, record: FeedbackRecord) -> _FTTTInstanceKey:
        return self._instance_key(prompt_idx=int(record.prompt_idx), request_id=record.request_id)

    def _state_for_key(self, key: _FTTTInstanceKey) -> FTTTInstanceState:
        return self._instances.get(key, FTTTInstanceState())

    def _aggregate_state_for_prompt(self, prompt_idx: int) -> FTTTInstanceState:
        prompt_idx = int(prompt_idx)
        legacy_key = self._instance_key(prompt_idx=prompt_idx)
        if legacy_key in self._instances:
            return self._instances[legacy_key]
        states = [state for key, state in self._instances.items() if int(key.prompt_idx) == prompt_idx]
        if not states:
            return FTTTInstanceState()
        return FTTTInstanceState(
            retries=max(int(state.retries) for state in states),
            stopped=any(bool(state.stopped) for state in states),
            adapter_version=max(int(state.adapter_version) for state in states),
        )

    def instance_state(self, prompt_idx: int, *, request_id: str = "") -> FTTTInstanceState:
        if request_id:
            return self._state_for_key(self._instance_key(prompt_idx=int(prompt_idx), request_id=request_id))
        return self._aggregate_state_for_prompt(int(prompt_idx))

    def reset_instance(self, prompt_idx: int, *, request_id: str = "") -> None:
        prompt_idx = int(prompt_idx)
        if request_id:
            self._instances.pop(self._instance_key(prompt_idx=prompt_idx, request_id=request_id), None)
            return
        for key in tuple(self._instances):
            if int(key.prompt_idx) == prompt_idx:
                self._instances.pop(key, None)

    def _lifecycle_decision(
        self,
        record: FeedbackRecord,
        state: FTTTInstanceState,
    ) -> _FTTTLifecycleDecision:
        if bool(state.stopped):
            return _FTTTLifecycleDecision(
                adaptable=False,
                next_state=state,
                retry_action="stop",
                retry_reason="instance-stopped",
                retry_after_update=False,
                barrier_state="blocked",
                barrier_reason="instance-stopped",
                ready_count=0,
            )
        if is_positive_feedback(record, positive_reward_threshold=float(self.config.positive_reward_threshold)):
            next_state = FTTTInstanceState(
                retries=int(state.retries),
                stopped=True,
                adapter_version=int(state.adapter_version),
            )
            return _FTTTLifecycleDecision(
                adaptable=False,
                next_state=next_state,
                retry_action="accept",
                retry_reason="positive-feedback",
                retry_after_update=False,
                barrier_state="blocked",
                barrier_reason="positive-feedback",
                ready_count=0,
            )
        next_retries = int(state.retries) + 1
        retry_limit_reached = int(self.config.max_retries) > 0 and next_retries >= int(self.config.max_retries)
        next_state = FTTTInstanceState(
            retries=next_retries,
            stopped=retry_limit_reached,
            adapter_version=int(state.adapter_version),
        )
        return _FTTTLifecycleDecision(
            adaptable=True,
            next_state=next_state,
            retry_action="stop" if retry_limit_reached else "retry",
            retry_reason="retry-limit" if retry_limit_reached else "feedback-update",
            retry_after_update=not retry_limit_reached,
            barrier_state="ready",
            barrier_reason="feedback-update-ready",
            ready_count=1,
        )

    def _record_is_adaptable(self, record: FeedbackRecord) -> bool:
        key = self._instance_key_for_record(record)
        decision = self._lifecycle_decision(record, self._state_for_key(key))
        self._instances[key] = decision.next_state
        if not decision.adaptable:
            self._skipped_feedback += 1
            return False
        return True

    def _mark_records_updated(self, feedback: Sequence[FeedbackRecord]) -> None:
        adapter_version = 0 if self._compiled is None else int(self._compiled.update_count)
        for record in feedback:
            key = self._instance_key_for_record(record)
            state = self._state_for_key(key)
            self._instances[key] = FTTTInstanceState(
                retries=int(state.retries),
                stopped=bool(state.stopped),
                adapter_version=adapter_version,
            )

    # -- trace judging and closed-loop intents (pure Python) ------------------

    def _toy_feedback_token_ids(self, feedback_text: str) -> tuple[int, ...]:
        tokens = re.findall(r"[a-z0-9_]+", feedback_text.lower())
        if not tokens:
            tokens = [feedback_text]
        out: list[int] = []
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            out.append(int.from_bytes(digest, byteorder="little", signed=False))
        return tuple(out)

    def _outcome_for_trace_entry(
        self,
        entry: CompletionTraceEntry,
        result: FTTTJudgeResult,
    ) -> RewardOutcomeEntry:
        feedback_text = result.feedback_text or ("correct" if result.is_correct else "Your answer is incorrect.")
        skip_reason = result.skip_reason
        if result.is_correct and not skip_reason:
            skip_reason = "correct-output"
        return RewardOutcomeEntry(
            request_id=entry.request_id,
            prompt_idx=entry.prompt_idx,
            sample_idx=entry.sample_idx,
            output_index=entry.output_index,
            group_id=entry.group_id,
            reward=result.reward,
            judge=result.judge,
            label=result.label,
            feedback_text=feedback_text,
            skip_reason=skip_reason,
            is_correct=result.is_correct,
        )

    def _feedback_record_for_trace_entry(
        self,
        entry: CompletionTraceEntry,
        outcome: RewardOutcomeEntry,
        result: FTTTJudgeResult,
    ) -> FeedbackRecord:
        feedback_text = outcome.feedback_text
        feedback_token_ids = result.feedback_token_ids or self._toy_feedback_token_ids(feedback_text)
        return FeedbackRecord(
            prompt_idx=int(entry.prompt_idx),
            feedback=feedback_text,
            reward=outcome.reward if outcome.reward is not None else 0.0,
            request_id=str(entry.request_id),
            sample_idx=int(entry.sample_idx),
            output_index=int(entry.output_index),
            group_id=str(entry.group_id),
            failed_text=str(entry.text),
            failed_token_ids=tuple(int(token_id) for token_id in entry.token_ids),
            feedback_token_ids=feedback_token_ids,
        )

    def closed_loop_intent_for_trace(self, trace: GenerationTraceBatch) -> FTTTClosedLoopIntent:
        if not isinstance(trace, GenerationTraceBatch):
            raise TypeError("trace must be GenerationTraceBatch")
        outcomes: list[RewardOutcomeEntry] = []
        retry_entries: list[RetryControlEntry] = []
        barrier_entries: list[UpdateBarrierEntry] = []
        feedback_records: list[FeedbackRecord] = []
        simulated_states: dict[_FTTTInstanceKey, FTTTInstanceState] = {}
        for trace_entry in trace.entries:
            result = self._judge(trace_entry)
            if not isinstance(result, FTTTJudgeResult):
                raise TypeError("FTTT judge must return FTTTJudgeResult")
            entry = self._outcome_for_trace_entry(trace_entry, result)
            outcomes.append(entry)
            if bool(entry.is_correct):
                retry_entries.append(
                    RetryControlEntry(
                        request_id=str(entry.request_id),
                        prompt_idx=int(entry.prompt_idx),
                        sample_idx=int(entry.sample_idx),
                        output_index=int(entry.output_index),
                        group_id=str(entry.group_id),
                        action="accept",
                        reason="correct-output",
                        retry_after_update=False,
                    )
                )
                ready_count = 0
                state = "blocked"
                reason = "correct-output"
            else:
                record = self._feedback_record_for_trace_entry(trace_entry, entry, result)
                key = self._instance_key_for_record(record)
                current_state = simulated_states.get(key, self._state_for_key(key))
                decision = self._lifecycle_decision(record, current_state)
                simulated_states[key] = decision.next_state
                if decision.adaptable:
                    feedback_records.append(record)
                retry_entries.append(
                    RetryControlEntry(
                        request_id=str(entry.request_id),
                        prompt_idx=int(entry.prompt_idx),
                        sample_idx=int(entry.sample_idx),
                        output_index=int(entry.output_index),
                        group_id=str(entry.group_id),
                        action=decision.retry_action,
                        reason=decision.retry_reason,
                        retry_after_update=decision.retry_after_update,
                    )
                )
                ready_count = decision.ready_count
                state = decision.barrier_state
                reason = decision.barrier_reason
            barrier_entries.append(
                UpdateBarrierEntry(
                    barrier_id=f"fttt:{entry.group_id}:{int(entry.prompt_idx)}:{int(entry.sample_idx)}",
                    group_id=str(entry.group_id),
                    consumer_id=str(self.consumer_id),
                    state=state,
                    request_ids=(str(entry.request_id),),
                    ready_count=ready_count,
                    reason=reason,
                )
            )
        return FTTTClosedLoopIntent(
            reward_outcomes=RewardOutcomeBatch(entries=tuple(outcomes)),
            retry_controls=RetryControlBatch(entries=tuple(retry_entries)),
            update_barriers=UpdateBarrierBatch(entries=tuple(barrier_entries)),
            feedback_records=tuple(feedback_records),
            update_count=len(feedback_records),
        )

    def consume_bundle(self, bundle, ctx) -> None:
        _ = ctx
        inputs = post_generate_input(bundle)
        intent = self.closed_loop_intent_for_trace(inputs.trace)
        inputs.reward_outcome.append(intent.reward_outcomes)
        for record in intent.feedback_records:
            self.enqueue_feedback(record)
        if inputs.retry_control is not None:
            inputs.retry_control.append(intent.retry_controls)
        if inputs.update_barrier is not None:
            inputs.update_barrier.append(intent.update_barriers)

    # -- tensor training and editing (compiled program) -----------------------

    def _ensure_compiled(self, hidden: torch.Tensor) -> None:
        if self._compiled is not None:
            return
        if hidden.ndim != 2:
            raise RuntimeError("FTTT update expects 2D hidden rows")
        hidden_size = int(hidden.shape[1])
        if int(self.config.hidden_size) > 0 and int(self.config.hidden_size) != hidden_size:
            raise RuntimeError("FTTT hidden size does not match configured hidden_size")
        program = FTTTProgram(
            hidden_size=hidden_size,
            feedback_feature_size=int(self.config.feedback_feature_size),
            rank=int(self.config.rank),
            lr=float(self.config.learning_rate),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        rows = 2  # export requires example rows strictly above the lower bound
        edit_input_size = hidden_size + max(1, int(self.config.feedback_feature_size)) + 1
        forward_ex = (
            torch.zeros((rows, edit_input_size), device=hidden.device, dtype=hidden.dtype),
            torch.zeros((rows, hidden_size), device=hidden.device, dtype=hidden.dtype),
        )
        predict_ex = (forward_ex[0],)
        row_dims = tuple(
            DynamicDim(value_path=(i,), dimension=0, lower_bound=1, upper_bound=65536)
            for i in range(2)
        )
        self._compiled = compile_program(
            program,
            states={
                "edit_adapter": StateSpec(scope=StateScope.REPLICA, optimizer=torch.optim.Adam)
            },
            examples={"forward": forward_ex, "predict": predict_ex},
            dynamic_dims={"forward": row_dims, "predict": row_dims[:1]},
        )

    def apply_feedback_update(
        self,
        hidden: torch.Tensor,
        feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord],
    ) -> None:
        if not bool(self.config.enabled):
            return
        records = list(feedback)
        if int(hidden.shape[0]) != len(records):
            raise RuntimeError("FTTT update requires one feedback record per hidden row")
        if not records:
            return
        self._ensure_compiled(hidden)
        assert self._compiled is not None
        adaptable_indices = [idx for idx, record in enumerate(records) if self._record_is_adaptable(record)]
        if not adaptable_indices:
            return
        index = torch.tensor(adaptable_indices, device=hidden.device, dtype=torch.long)
        train_feedback = [records[idx] for idx in adaptable_indices]
        train_hidden = hidden.detach().index_select(0, index)
        edit_input = edit_inputs(
            train_hidden, train_feedback, feature_size=int(self.config.feedback_feature_size)
        )
        target = target_delta(
            train_hidden,
            train_feedback,
            feature_size=int(self.config.feedback_feature_size),
            edit_scale=float(self.config.edit_scale),
        )
        self._step_id += 1
        bundle = provided_bundle(
            consumer_id=self.consumer_id,
            step_id=self._step_id,
            entries={"edit_input": edit_input, "target": target},
        )
        self._compiled.consume_flow(self._compiled.flow_ordinal("forward"), bundle, None)
        self._feedback_count += len(train_feedback)
        loss_sum = float(self._compiled.loss_sum)
        self._last_loss = loss_sum - self._loss_sum_seen
        self._loss_sum_seen = loss_sum
        self._mark_records_updated(train_feedback)

    def edit_delta(
        self,
        hidden: torch.Tensor,
        feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord],
    ) -> torch.Tensor:
        if not bool(self.config.enabled):
            return torch.zeros_like(hidden)
        records = list(feedback)
        if not records:
            return torch.zeros_like(hidden)
        self._ensure_compiled(hidden)
        assert self._compiled is not None
        edit_input = edit_inputs(
            hidden.detach(), records, feature_size=int(self.config.feedback_feature_size)
        )
        delta = self._compiled.run_flow("predict", edit_input)["delta"]
        masks = [
            0.0 if self._state_for_key(self._instance_key_for_record(record)).stopped else 1.0
            for record in records
        ]
        mask = torch.tensor(masks, device=hidden.device, dtype=hidden.dtype).reshape(-1, 1)
        return delta * mask

    def apply_edit_to_hidden(
        self,
        hidden: torch.Tensor,
        feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord],
    ) -> torch.Tensor:
        return hidden + self.edit_delta(hidden, feedback)

    def apply_edit_to_logits(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord],
        *,
        output_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = self.edit_delta(hidden, feedback)
        if output_weight is not None:
            logit_delta = F.linear(delta, output_weight.to(device=delta.device, dtype=delta.dtype))
        else:
            if int(logits.shape[-1]) != int(delta.shape[-1]):
                raise RuntimeError("FTTT logits edit without output_weight requires logits and hidden to share width")
            logit_delta = delta
        return logits + logit_delta.to(device=logits.device, dtype=logits.dtype)

    def read_stats(self) -> FTTTStats:
        compiled = self._compiled
        update_count = 0 if compiled is None else int(compiled.update_count)
        return FTTTStats(
            queued_feedback=len(self._queue),
            dropped_feedback=int(self._dropped_feedback),
            feedback_count=int(self._feedback_count),
            update_count=update_count,
            loss_count=0 if compiled is None else int(compiled.loss_count),
            last_loss=float(self._last_loss),
            skipped_feedback=int(self._skipped_feedback),
            stopped_instances=sum(1 for state in self._instances.values() if state.stopped),
            adapter_version=update_count,
        )


__all__ = ["FTTTClosedLoopIntent", "FTTTConsumer", "FTTTStats"]
