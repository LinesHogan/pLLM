"""Transactional typed judge/feedback service for canonical FTTT."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from hashlib import blake2b
import re

import torch

from tllm.canonical.host_runtime import HostReceipt, NoHostRequest
from tllm.canonical.workloads.fttt.judge import FTTTJudgeService
from tllm.canonical.workloads.fttt.types import (
    FTTTBarrierState,
    FTTTClosedLoopIntent,
    FTTTCompletionBatch,
    FTTTConfig,
    FTTTControlReason,
    FTTTCpuReadiness,
    FTTTCudaReadiness,
    FTTTFeedbackRecord,
    FTTTHostTransaction,
    FTTTInstanceKey,
    FTTTInstanceState,
    FTTTJudgeResult,
    FTTTLifecycleTransition,
    FTTTRewardOutcome,
    FTTTResult,
    FTTTRetryAction,
    FTTTRetryControl,
    FTTTTrainingPlan,
    FTTTUpdateBarrier,
)


@dataclass(frozen=True, slots=True)
class _LifecycleDecision:
    adaptable: bool
    next_state: FTTTInstanceState
    action: FTTTRetryAction
    reason: FTTTControlReason
    retry_after_update: bool
    barrier_state: FTTTBarrierState
    ready_count: int


@dataclass(frozen=True, slots=True)
class _NoActiveTransaction:
    pass


@dataclass(frozen=True, slots=True)
class _ActiveTransaction:
    transaction: FTTTHostTransaction


_ActiveState = _NoActiveTransaction | _ActiveTransaction


class FTTTFeedbackService:
    """Own judge I/O, bounded queue, and lifecycle settlement only."""

    __slots__ = (
        "_active",
        "_config",
        "_dropped_feedback",
        "_instances",
        "_judge",
        "_next_transaction_id",
        "_projected_instances",
        "_queue",
        "_queued_feedback",
    )

    def __init__(self, config: FTTTConfig, judge: FTTTJudgeService) -> None:
        if type(config) is not FTTTConfig:
            raise TypeError("FTTT feedback service requires exact config")
        self._config = config
        self._judge = judge
        self._instances: dict[FTTTInstanceKey, FTTTInstanceState] = {}
        self._projected_instances: dict[FTTTInstanceKey, FTTTInstanceState] = {}
        self._queue: deque[FTTTHostTransaction] = deque()
        self._active: _ActiveState = _NoActiveTransaction()
        self._next_transaction_id = 1
        self._queued_feedback = 0
        self._dropped_feedback = 0

    @property
    def queued_feedback(self) -> int:
        return self._queued_feedback

    @property
    def queued_transactions(self) -> int:
        return len(self._queue)

    @property
    def dropped_feedback(self) -> int:
        return self._dropped_feedback

    @property
    def config(self) -> FTTTConfig:
        return self._config

    def instance_state(self, key: FTTTInstanceKey) -> FTTTInstanceState:
        return self._instances.get(key, FTTTInstanceState())

    def enqueue(
        self,
        completions: FTTTCompletionBatch,
        hidden_rows: torch.Tensor,
    ) -> bool:
        if (
            type(completions) is not FTTTCompletionBatch
            or type(hidden_rows) is not torch.Tensor
            or hidden_rows.ndim != 2
            or int(hidden_rows.shape[0]) != len(completions.entries)
            or hidden_rows.device.type != "cpu"
            or hidden_rows.dtype is not torch.float32
        ):
            raise ValueError(
                "FTTT CPU host input requires one float32 CPU row per completion"
            )
        with torch.inference_mode(False):
            owned_hidden_rows = hidden_rows.detach().clone()
        if (
            torch.is_inference(owned_hidden_rows)
        ):
            raise RuntimeError("FTTT CPU host submission did not create an owned tensor")
        return self._enqueue_owned(
            completions,
            owned_hidden_rows,
            FTTTCpuReadiness(),
        )

    def enqueue_cuda(
        self,
        completions: FTTTCompletionBatch,
        hidden_rows: torch.Tensor,
        producer_stream: torch.cuda.Stream,
    ) -> bool:
        if (
            type(completions) is not FTTTCompletionBatch
            or type(hidden_rows) is not torch.Tensor
            or hidden_rows.ndim != 2
            or int(hidden_rows.shape[0]) != len(completions.entries)
            or hidden_rows.device.type != "cuda"
            or hidden_rows.dtype is not torch.float32
            or type(producer_stream) is not torch.cuda.Stream
            or producer_stream.device != hidden_rows.device
        ):
            raise ValueError(
                "FTTT CUDA host input requires one float32 row and its producer stream"
            )
        with torch.cuda.stream(producer_stream), torch.inference_mode(False):
            owned_hidden_rows = hidden_rows.detach().clone()
            readiness = FTTTCudaReadiness(producer_stream)
        if (
            torch.is_inference(owned_hidden_rows)
        ):
            raise RuntimeError("FTTT CUDA host submission did not create an owned tensor")
        return self._enqueue_owned(completions, owned_hidden_rows, readiness)

    def _enqueue_owned(
        self,
        completions: FTTTCompletionBatch,
        owned_hidden_rows: torch.Tensor,
        readiness: FTTTCpuReadiness | FTTTCudaReadiness,
    ) -> bool:
        outcomes: list[FTTTRewardOutcome] = []
        retry_controls: list[FTTTRetryControl] = []
        barriers: list[FTTTUpdateBarrier] = []
        records: list[FTTTFeedbackRecord] = []
        transitions: list[FTTTLifecycleTransition] = []
        selected_rows: list[int] = []
        available = self._config.max_queue_size - self._queued_feedback
        projected_instances = dict(self._projected_instances)
        dropped_feedback = self._dropped_feedback
        for row, completion in enumerate(completions.entries):
            judged = self._judge.judge(completion)
            if type(judged) is not FTTTJudgeResult:
                raise TypeError("FTTT judge must return an exact FTTTJudgeResult")
            feedback_text = judged.feedback_text or (
                "correct" if judged.is_correct else "Your answer is incorrect."
            )
            outcome = FTTTRewardOutcome(
                completion,
                judged.reward,
                judged.is_correct,
                feedback_text,
                judged.judge,
                judged.label,
                judged.skip_reason
                or (
                    FTTTControlReason.CORRECT_OUTPUT.value
                    if judged.is_correct
                    else ""
                ),
            )
            outcomes.append(outcome)
            key = FTTTInstanceKey(
                completion.identity.request_id,
                completion.identity.prompt_idx,
            )
            previous = projected_instances.get(
                key,
                self._instances.get(key, FTTTInstanceState()),
            )
            if judged.is_correct:
                decision = _LifecycleDecision(
                    False,
                    replace(previous, stopped=True),
                    FTTTRetryAction.ACCEPT,
                    FTTTControlReason.CORRECT_OUTPUT,
                    False,
                    FTTTBarrierState.BLOCKED,
                    0,
                )
                trained = False
            else:
                record = FTTTFeedbackRecord(
                    completion,
                    feedback_text,
                    judged.feedback_token_ids
                    or self._feedback_token_ids(feedback_text),
                    judged.reward,
                )
                decision = self._decision(record, previous)
                trained = decision.adaptable and len(records) < available
                if trained:
                    records.append(record)
                    selected_rows.append(row)
                elif decision.adaptable:
                    dropped_feedback += 1
                    decision = _LifecycleDecision(
                        False,
                        previous,
                        FTTTRetryAction.STOP,
                        FTTTControlReason.FEEDBACK_QUEUE_FULL,
                        False,
                        FTTTBarrierState.BLOCKED,
                        0,
                    )
            transitions.append(
                FTTTLifecycleTransition(
                    key,
                    previous,
                    decision.next_state,
                    trained,
                )
            )
            projected_instances[key] = decision.next_state
            retry_controls.append(
                FTTTRetryControl(
                    completion,
                    decision.action,
                    decision.reason,
                    decision.retry_after_update,
                )
            )
            barriers.append(
                FTTTUpdateBarrier(
                    "fttt:"
                    + completion.group_id
                    + ":"
                    + str(completion.identity.prompt_idx)
                    + ":"
                    + str(completion.identity.sample_idx),
                    completion.group_id,
                    decision.barrier_state,
                    (completion.identity.request_id,),
                    decision.ready_count,
                    decision.reason,
                )
            )
        intent = FTTTClosedLoopIntent(
            self._next_transaction_id,
            tuple(outcomes),
            tuple(retry_controls),
            tuple(barriers),
            tuple(records),
            tuple(transitions),
        )
        transaction = FTTTHostTransaction(
            owned_hidden_rows,
            FTTTTrainingPlan(tuple(selected_rows), intent),
            readiness,
        )
        self._queue.append(transaction)
        self._projected_instances = projected_instances
        self._dropped_feedback = dropped_feedback
        self._next_transaction_id += 1
        self._queued_feedback += len(records)
        return bool(records)

    def begin(self, request: NoHostRequest) -> FTTTHostTransaction | None:
        del request
        match self._active:
            case _NoActiveTransaction():
                pass
            case _ActiveTransaction():
                raise RuntimeError("active FTTT transaction is unresolved")
        if not self._queue:
            return None
        transaction = self._queue.popleft()
        self._queued_feedback -= len(transaction.plan.intent.feedback_records)
        self._active = _ActiveTransaction(transaction)
        return transaction

    def commit(
        self,
        transaction: FTTTHostTransaction,
        receipt: HostReceipt[FTTTResult],
    ) -> None:
        intent = receipt.result.intent
        match self._active:
            case _ActiveTransaction(transaction=active) if active is transaction:
                pass
            case _NoActiveTransaction() | _ActiveTransaction():
                raise RuntimeError("FTTT service has no active transaction")
        expected = replace(
            active.plan.intent,
            adapter_version=intent.adapter_version,
        )
        if intent != expected:
            raise RuntimeError("FTTT execution rewrote the host-service intent")
        settled = dict(self._instances)
        for transition in intent.transitions:
            current = settled.get(transition.key, FTTTInstanceState())
            if (
                current.retries != transition.previous.retries
                or current.stopped != transition.previous.stopped
            ):
                raise RuntimeError("FTTT lifecycle transition is stale")
            version = (
                intent.adapter_version
                if transition.trained
                else current.adapter_version
            )
            settled[transition.key] = replace(
                transition.next,
                adapter_version=version,
            )
        projected = self._projected_instances if self._queue else dict(settled)
        self._instances = settled
        self._active = _NoActiveTransaction()
        self._projected_instances = projected

    def abort(self, transaction: FTTTHostTransaction) -> None:
        match self._active:
            case _ActiveTransaction(transaction=active) if active is transaction:
                pass
            case _NoActiveTransaction() | _ActiveTransaction():
                raise RuntimeError("FTTT abort does not match active transaction")
        self._active = _NoActiveTransaction()
        self._queue.clear()
        self._queued_feedback = 0
        self._projected_instances = dict(self._instances)

    def _decision(
        self,
        record: FTTTFeedbackRecord,
        current: FTTTInstanceState,
    ) -> _LifecycleDecision:
        if current.stopped:
            return _LifecycleDecision(
                False,
                current,
                FTTTRetryAction.STOP,
                FTTTControlReason.INSTANCE_STOPPED,
                False,
                FTTTBarrierState.BLOCKED,
                0,
            )
        tokens = set(re.findall(r"[a-z0-9_]+", record.feedback_text.lower()))
        negative = {
            "incorrect",
            "wrong",
            "bad",
            "fail",
            "failed",
            "failure",
            "mistake",
            "error",
        }
        positive = {"correct", "right", "solved", "passed", "positive"}
        if (
            record.reward >= self._config.positive_reward_threshold
            or bool(tokens & positive) and not bool(tokens & negative)
        ):
            return _LifecycleDecision(
                False,
                replace(current, stopped=True),
                FTTTRetryAction.ACCEPT,
                FTTTControlReason.POSITIVE_FEEDBACK,
                False,
                FTTTBarrierState.BLOCKED,
                0,
            )
        retries = current.retries + 1
        stopped = (
            self._config.max_retries > 0
            and retries >= self._config.max_retries
        )
        return _LifecycleDecision(
            True,
            FTTTInstanceState(retries, stopped, current.adapter_version),
            FTTTRetryAction.STOP if stopped else FTTTRetryAction.RETRY,
            FTTTControlReason.RETRY_LIMIT
            if stopped
            else FTTTControlReason.FEEDBACK_UPDATE,
            not stopped,
            FTTTBarrierState.READY,
            1,
        )

    @staticmethod
    def _feedback_token_ids(text: str) -> tuple[int, ...]:
        tokens = re.findall(r"[a-z0-9_]+", text.lower()) or (text,)
        return tuple(
            int.from_bytes(
                blake2b(token.encode("utf-8"), digest_size=4).digest(),
                byteorder="little",
            )
            for token in tokens
        )


__all__ = [
    "FTTTFeedbackService",
]
