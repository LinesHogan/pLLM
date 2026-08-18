#!/usr/bin/env python3
"""FTTT-inspired feedback edit consumer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Sequence

import torch

from tllm.contracts.test_time_lora import LoRAScope, TestTimeLoRAUpdate
from tllm.contracts.test_time_training import (
    TestTimeTrainerRequest,
    TestTimeTrainerResult,
    TestTimeTrainingBatch,
    TestTimeTrainingExample,
)
from tllm.consumers.base import BaseConsumer
from tllm.consumers._common.bundle_inputs import post_generate_input
from tllm.consumers.fttt.config import FTTTConsumerConfig
from tllm.consumers.fttt.engine import (
    FeedbackRecord,
    FTTTEngine,
    FTTTInstanceState,
    FTTTJudge,
    FTTTJudgeResult,
    FTTTStats,
    _default_trace_judge,
    _FTTTInstanceKey,
    _FTTTLifecycleDecision,
)
from tllm.ports.generation_trace import GenerationTrace
from tllm.ports.base import ConsumerFlow
from tllm.ports.reward_outcome import RewardOutcome
from tllm.ports.retry_control import RetryControl
from tllm.ports.update_barrier import UpdateBarrier
from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch, RewardOutcomeEntry
from tllm.runtime.retry_control_runtime import RetryControlBatch, RetryControlEntry
from tllm.runtime.test_time.lora import TestTimeLoRAService
from tllm.runtime.test_time.trainer import TestTimeTrainerBridge, stage_trainer_result_updates
from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch, UpdateBarrierEntry


@dataclass(frozen=True)
class FTTTClosedLoopIntent:
    reward_outcomes: RewardOutcomeBatch
    retry_controls: RetryControlBatch
    update_barriers: UpdateBarrierBatch
    feedback_records: tuple[FeedbackRecord, ...]
    update_count: int


@dataclass(frozen=True)
class FTTTToyClosedLoopResult:
    reward_outcomes: RewardOutcomeBatch
    retry_controls: RetryControlBatch
    update_barriers: UpdateBarrierBatch
    feedback_records: tuple[FeedbackRecord, ...]
    update_count: int
    adapter_version: int
    instance_states: dict[int, FTTTInstanceState]
    lora_update: TestTimeLoRAUpdate | None = None


@dataclass(frozen=True)
class FTTTTrainerBridgeClosedLoopResult:
    reward_outcomes: RewardOutcomeBatch
    retry_controls: RetryControlBatch
    update_barriers: UpdateBarrierBatch
    feedback_records: tuple[FeedbackRecord, ...]
    update_count: int
    training_batch: TestTimeTrainingBatch | None = None
    trainer_request: TestTimeTrainerRequest | None = None
    trainer_result: TestTimeTrainerResult | None = None
    staged_update_count: int = 0


class FTTTConsumer(BaseConsumer[FTTTConsumerConfig]):
    """Feedback-conditioned local edit consumer."""

    def __init__(
        self,
        config: FTTTConsumerConfig,
        *,
        engine: FTTTEngine | None = None,
        judge: FTTTJudge | None = None,
    ) -> None:
        self.config = config
        self.engine = engine or FTTTEngine(config)
        self._judge = judge or _default_trace_judge
        self._queue: deque[FeedbackRecord] = deque()
        self._dropped_feedback = 0

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

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

    def enqueue_feedback(self, feedback: FeedbackRecord) -> bool:
        if len(self._queue) >= max(0, int(self.config.max_queue_size)):
            self._dropped_feedback += 1
            return False
        self._queue.append(feedback)
        return True

    def drain_feedback(self, *, limit: int = 0) -> list[FeedbackRecord]:
        max_items = len(self._queue) if int(limit) <= 0 else min(int(limit), len(self._queue))
        return [self._queue.popleft() for _ in range(max_items)]

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

    def _toy_feedback_token_ids(self, feedback_text: str) -> tuple[int, ...]:
        tokens = re.findall(r"[a-z0-9_]+", feedback_text.lower())
        if not tokens:
            tokens = [feedback_text]
        out: list[int] = []
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
            out.append(int.from_bytes(digest, byteorder="little", signed=False))
        return tuple(out)

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

    def process_generation_trace(
        self,
        trace: GenerationTraceBatch,
        *,
        judge: FTTTJudge | None = None,
        enqueue_failures: bool = True,
    ) -> RewardOutcomeBatch:
        if not isinstance(trace, GenerationTraceBatch):
            raise TypeError("trace must be GenerationTraceBatch")
        judge_fn = judge or self._judge
        outcomes: list[RewardOutcomeEntry] = []
        for entry in trace.entries:
            result = judge_fn(entry)
            if not isinstance(result, FTTTJudgeResult):
                raise TypeError("FTTT judge must return FTTTJudgeResult")
            outcome = self._outcome_for_trace_entry(entry, result)
            outcomes.append(outcome)
            if enqueue_failures and result.is_correct is False:
                self.enqueue_feedback(self._feedback_record_for_trace_entry(entry, outcome, result))
        return RewardOutcomeBatch(entries=tuple(outcomes))

    def _retry_entry_for_lifecycle_decision(
        self,
        entry: RewardOutcomeEntry,
        decision: _FTTTLifecycleDecision,
    ) -> RetryControlEntry:
        return RetryControlEntry(
            request_id=str(entry.request_id),
            prompt_idx=int(entry.prompt_idx),
            sample_idx=int(entry.sample_idx),
            output_index=int(entry.output_index),
            group_id=str(entry.group_id),
            action=decision.retry_action,
            reason=decision.retry_reason,
            retry_after_update=decision.retry_after_update,
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
                key = self.engine._instance_key_for_record(record)
                current_state = simulated_states.get(key, self.engine._state_for_key(key))
                decision = self.engine._lifecycle_decision(record, current_state)
                simulated_states[key] = decision.next_state
                if decision.adaptable:
                    feedback_records.append(record)
                retry_entries.append(self._retry_entry_for_lifecycle_decision(entry, decision))
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

    def _retry_controls_after_toy_update(
        self,
        batch: RetryControlBatch,
        *,
        adapter_version: int,
    ) -> RetryControlBatch:
        entries: list[RetryControlEntry] = []
        for entry in batch.entries:
            reason = entry.reason
            if entry.retry_after_update:
                suffix = f"adapter-version={int(adapter_version)}"
                reason = f"{reason};{suffix}" if reason else suffix
            entries.append(
                RetryControlEntry(
                    request_id=entry.request_id,
                    prompt_idx=entry.prompt_idx,
                    sample_idx=entry.sample_idx,
                    output_index=entry.output_index,
                    group_id=entry.group_id,
                    action=entry.action,
                    reason=reason,
                    retry_after_update=entry.retry_after_update,
                )
            )
        return RetryControlBatch(entries=tuple(entries))

    def feedback_training_batch(
        self,
        feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord],
    ) -> TestTimeTrainingBatch:
        records = tuple(feedback)
        examples: list[TestTimeTrainingExample] = []
        for record in records:
            if not isinstance(record, FeedbackRecord):
                raise TypeError("feedback must contain FeedbackRecord")
            if not record.request_id:
                raise ValueError("feedback training batch requires request_id")
            feedback_token_ids = record.feedback_token_ids or self._toy_feedback_token_ids(record.feedback)
            token_ids = tuple(record.failed_token_ids) + tuple(feedback_token_ids)
            labels = tuple(-100 for _ in record.failed_token_ids) + tuple(feedback_token_ids)
            examples.append(
                TestTimeTrainingExample(
                    request_id=record.request_id,
                    prompt_id=str(int(record.prompt_idx)),
                    sample_id=str(int(record.sample_idx)),
                    output_id=str(int(record.output_index)),
                    objective_kind="feedback_lm",
                    token_ids=token_ids,
                    labels=labels,
                    rewards=(float(record.reward),),
                    metadata={
                        "consumer_id": self.consumer_id,
                        "feedback_text": record.feedback,
                        "failed_text": record.failed_text,
                        "failed_token_ids": tuple(record.failed_token_ids),
                        "feedback_token_ids": tuple(feedback_token_ids),
                        "group_id": record.group_id,
                    },
                )
            )
        return TestTimeTrainingBatch(
            examples=tuple(examples),
            metadata={
                "consumer_id": self.consumer_id,
                "objective_kind": "feedback_lm",
            },
        )

    def submit_feedback_training_batch(
        self,
        batch: TestTimeTrainingBatch,
        *,
        trainer_bridge: TestTimeTrainerBridge,
        adapter_key: str,
        target_modules: tuple[str, ...],
        rank: int,
        scope: LoRAScope = "request",
        request_ids: tuple[str, ...] | None = None,
        lora_service: TestTimeLoRAService | None = None,
        algorithm_id: str = "fttt",
        metadata: dict[str, object] | None = None,
    ) -> tuple[TestTimeTrainerRequest, TestTimeTrainerResult, int]:
        if not isinstance(batch, TestTimeTrainingBatch):
            raise TypeError("batch must be TestTimeTrainingBatch")
        train = getattr(trainer_bridge, "train", None)
        if not callable(train):
            raise TypeError("trainer_bridge must provide train")
        if request_ids is None:
            request_ids = batch.request_ids if scope == "request" else ()
        request = TestTimeTrainerRequest(
            algorithm_id=algorithm_id,
            adapter_key=adapter_key,
            target_modules=target_modules,
            rank=rank,
            scope=scope,
            request_ids=tuple(request_ids),
            examples=batch,
            metadata=metadata or {},
        )
        result = train(request)
        if not isinstance(result, TestTimeTrainerResult):
            raise TypeError("trainer bridge must return TestTimeTrainerResult")
        staged_count = 0
        if lora_service is not None:
            staged_count = stage_trainer_result_updates(request, result, lora_service)
        return request, result, staged_count

    def _adapter_key_for_feedback_training(
        self,
        feedback_records: Sequence[FeedbackRecord],
        *,
        adapter_key_prefix: str,
        step_id: int,
    ) -> str:
        first_record = feedback_records[0]
        group_id = first_record.group_id or "ungrouped"
        version = int(step_id) if int(step_id) >= 0 else int(self.engine.adapter_version) + 1
        return f"{adapter_key_prefix}:{group_id}:{int(first_record.prompt_idx)}:v{version}"

    def _retry_controls_after_trainer_update(
        self,
        batch: RetryControlBatch,
        result: TestTimeTrainerResult | None,
        *,
        staged_update_count: int,
    ) -> RetryControlBatch:
        updates = () if result is None else result.updates
        first_update = updates[0] if updates else None
        entries: list[RetryControlEntry] = []
        for entry in batch.entries:
            reason = entry.reason
            retry_after_update = bool(entry.retry_after_update)
            if entry.retry_after_update:
                if staged_update_count > 0 and first_update is not None:
                    suffix = (
                        f"adapter-key={first_update.adapter_key};"
                        f"adapter-version={int(first_update.step_id)}"
                    )
                    reason = f"{reason};{suffix}" if reason else suffix
                else:
                    retry_after_update = False
                    suffix = "trainer-no-update"
                    reason = f"{reason};{suffix}" if reason else suffix
            entries.append(
                RetryControlEntry(
                    request_id=entry.request_id,
                    prompt_idx=entry.prompt_idx,
                    sample_idx=entry.sample_idx,
                    output_index=entry.output_index,
                    group_id=entry.group_id,
                    action=entry.action,
                    reason=reason,
                    retry_after_update=retry_after_update,
                )
            )
        return RetryControlBatch(entries=tuple(entries))

    def _update_barriers_after_trainer_update(
        self,
        batch: UpdateBarrierBatch,
        *,
        staged_update_count: int,
    ) -> UpdateBarrierBatch:
        if staged_update_count > 0:
            return batch
        entries: list[UpdateBarrierEntry] = []
        for entry in batch.entries:
            if entry.state == "ready":
                reason = f"{entry.reason};trainer-no-update" if entry.reason else "trainer-no-update"
                entries.append(
                    UpdateBarrierEntry(
                        barrier_id=entry.barrier_id,
                        group_id=entry.group_id,
                        consumer_id=entry.consumer_id,
                        state="blocked",
                        request_ids=entry.request_ids,
                        ready_count=0,
                        reason=reason,
                    )
                )
            else:
                entries.append(entry)
        return UpdateBarrierBatch(entries=tuple(entries))

    def run_trainer_bridge_closed_loop_step(
        self,
        trace: GenerationTraceBatch,
        *,
        trainer_bridge: TestTimeTrainerBridge,
        target_modules: tuple[str, ...],
        rank: int,
        lora_service: TestTimeLoRAService | None = None,
        adapter_key_prefix: str = "fttt",
        step_id: int = -1,
        algorithm_id: str = "fttt",
    ) -> FTTTTrainerBridgeClosedLoopResult:
        intent = self.closed_loop_intent_for_trace(trace)
        if not intent.feedback_records:
            return FTTTTrainerBridgeClosedLoopResult(
                reward_outcomes=intent.reward_outcomes,
                retry_controls=intent.retry_controls,
                update_barriers=intent.update_barriers,
                feedback_records=intent.feedback_records,
                update_count=intent.update_count,
            )
        training_batch = self.feedback_training_batch(intent.feedback_records)
        adapter_key = self._adapter_key_for_feedback_training(
            intent.feedback_records,
            adapter_key_prefix=adapter_key_prefix,
            step_id=step_id,
        )
        trainer_request, trainer_result, staged_count = self.submit_feedback_training_batch(
            training_batch,
            trainer_bridge=trainer_bridge,
            adapter_key=adapter_key,
            target_modules=target_modules,
            rank=rank,
            scope="request",
            lora_service=lora_service,
            algorithm_id=algorithm_id,
            metadata={
                "consumer_id": self.consumer_id,
                "step_id": int(step_id),
            },
        )
        retry_controls = self._retry_controls_after_trainer_update(
            intent.retry_controls,
            trainer_result,
            staged_update_count=staged_count,
        )
        update_barriers = self._update_barriers_after_trainer_update(
            intent.update_barriers,
            staged_update_count=staged_count,
        )
        return FTTTTrainerBridgeClosedLoopResult(
            reward_outcomes=intent.reward_outcomes,
            retry_controls=retry_controls,
            update_barriers=update_barriers,
            feedback_records=intent.feedback_records,
            update_count=intent.update_count,
            training_batch=training_batch,
            trainer_request=trainer_request,
            trainer_result=trainer_result,
            staged_update_count=staged_count,
        )

    def run_toy_closed_loop_step(
        self,
        trace: GenerationTraceBatch,
        hidden: torch.Tensor,
        *,
        target_module_name: str | None = None,
        lora_int_id: int | None = None,
        adapter_key_prefix: str = "fttt",
        alpha: float = 1.0,
        step_id: int = -1,
    ) -> FTTTToyClosedLoopResult:
        intent = self.closed_loop_intent_for_trace(trace)
        if intent.feedback_records:
            self.apply_feedback_update(hidden, intent.feedback_records)
        adapter_version = self.engine.adapter_version
        retry_controls = self._retry_controls_after_toy_update(
            intent.retry_controls,
            adapter_version=adapter_version,
        )
        lora_update: TestTimeLoRAUpdate | None = None
        if target_module_name is not None or lora_int_id is not None:
            if target_module_name is None or lora_int_id is None:
                raise ValueError("target_module_name and lora_int_id must be provided together")
            if not intent.feedback_records:
                raise RuntimeError("cannot export FTTT LoRA update without feedback records")
            first_record = intent.feedback_records[0]
            group_id = first_record.group_id or "ungrouped"
            adapter_key = f"{adapter_key_prefix}:{group_id}:{int(first_record.prompt_idx)}:v{int(adapter_version)}"
            request_ids = tuple(dict.fromkeys(record.request_id for record in intent.feedback_records if record.request_id))
            lora_update = self.export_test_time_lora_update(
                adapter_key=adapter_key,
                lora_int_id=lora_int_id,
                target_module_name=target_module_name,
                rank=int(self.config.rank),
                scope="request",
                request_ids=request_ids,
                step_id=step_id,
                alpha=alpha,
            )
        prompt_indices = {int(record.prompt_idx) for record in intent.feedback_records}
        return FTTTToyClosedLoopResult(
            reward_outcomes=intent.reward_outcomes,
            retry_controls=retry_controls,
            update_barriers=intent.update_barriers,
            feedback_records=intent.feedback_records,
            update_count=intent.update_count,
            adapter_version=adapter_version,
            instance_states={prompt_idx: self.instance_state(prompt_idx) for prompt_idx in prompt_indices},
            lora_update=lora_update,
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

    def apply_feedback_update(self, hidden: torch.Tensor, feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord]) -> None:
        records = list(feedback)
        self.engine.update(hidden, records)

    def edit_delta(self, hidden: torch.Tensor, feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord]) -> torch.Tensor:
        records = list(feedback)
        return self.engine.edit_delta(hidden, records)

    def apply_edit_to_hidden(
        self,
        hidden: torch.Tensor,
        feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord],
    ) -> torch.Tensor:
        records = list(feedback)
        return self.engine.apply_edit_to_hidden(hidden, records)

    def apply_edit_to_logits(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord],
        *,
        output_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        records = list(feedback)
        return self.engine.apply_edit_to_logits(logits, hidden, records, output_weight=output_weight)

    def instance_state(self, prompt_idx: int, *, request_id: str = "", sample_idx: int = 0) -> FTTTInstanceState:
        return self.engine.instance_state(prompt_idx, request_id=request_id, sample_idx=sample_idx)

    def reset_instance(self, prompt_idx: int, *, request_id: str = "", sample_idx: int = 0) -> None:
        self.engine.reset_instance(prompt_idx, request_id=request_id, sample_idx=sample_idx)

    def read_stats(self) -> FTTTStats:
        return self.engine.read_stats(queued_feedback=len(self._queue), dropped_feedback=self._dropped_feedback)

    def export_test_time_lora_update(
        self,
        *,
        adapter_key: str,
        lora_int_id: int,
        target_module_name: str,
        rank: int,
        scope: LoRAScope = "global",
        request_ids: tuple[str, ...] = (),
        step_id: int = -1,
        alpha: float = 1.0,
    ) -> TestTimeLoRAUpdate:
        return self.engine.export_test_time_lora_update(
            adapter_key=adapter_key,
            lora_int_id=lora_int_id,
            target_module_name=target_module_name,
            rank=rank,
            scope=scope,
            request_ids=request_ids,
            step_id=step_id,
            alpha=alpha,
        )
