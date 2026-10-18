"""Transactional typed judge/feedback service for canonical FTTT."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, replace
from hashlib import blake2b

from pllm.contracts.feedback import BarrierState, ControlReason, RetryAction, RetryControl, RewardOutcome, UpdateBarrier
from pllm.execution.host_runtime import CompletionInput, HostReceipt, NoHostRequest
from pllm.workloads.fttt.judge import FTTTJudgeService
from pllm.workloads.fttt.types import (
    FTTTClosedLoopIntent,
    FTTTConfig,
    FTTTFeedbackRecord,
    FTTTHostTransaction,
    FTTTInstanceState,
    FTTTLifecycleTransition,
    FTTTResult,
)


@dataclass(frozen=True, slots=True)
class _LifecycleDecision:
    adaptable: bool
    next_state: FTTTInstanceState
    action: RetryAction
    reason: ControlReason


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
        self._config = config
        self._judge = judge
        self._instances: dict[tuple[str, int], FTTTInstanceState] = {}
        self._projected_instances: dict[tuple[str, int], FTTTInstanceState] = {}
        self._queue: deque[FTTTHostTransaction] = deque()
        self._active: FTTTHostTransaction | None = None
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

    def instance_state(self, group_id: str, prompt_idx: int) -> FTTTInstanceState:
        return self._instances.get((group_id, prompt_idx), FTTTInstanceState())

    def enqueue(self, batch: CompletionInput) -> bool:
        outcomes: list[RewardOutcome] = []
        retry_controls: list[RetryControl] = []
        barriers: list[UpdateBarrier] = []
        records: list[FTTTFeedbackRecord] = []
        transitions: list[FTTTLifecycleTransition] = []
        selected_rows: list[int] = []
        available = self._config.max_queue_size - self._queued_feedback
        projected_instances = dict(self._projected_instances)
        dropped_feedback = self._dropped_feedback
        for row, completion in enumerate(batch.completions.entries):
            judged = self._judge.judge(completion)
            feedback_text = judged.feedback_text or ("correct" if judged.is_correct else "Your answer is incorrect.")
            outcome = RewardOutcome(
                completion,
                judged.reward,
                judged.is_correct,
                feedback_text,
                judged.judge,
                judged.label,
                judged.skip_reason or (ControlReason.CORRECT_OUTPUT.value if judged.is_correct else ""),
            )
            outcomes.append(outcome)
            key = (completion.group_id, completion.identity.prompt_idx)
            previous = projected_instances.get(
                key,
                self._instances.get(key, FTTTInstanceState()),
            )
            if judged.is_correct:
                decision = _LifecycleDecision(
                    False,
                    replace(previous, stopped=True),
                    RetryAction.ACCEPT,
                    ControlReason.CORRECT_OUTPUT,
                )
                trained = False
            else:
                record = FTTTFeedbackRecord(
                    completion,
                    feedback_text,
                    judged.feedback_token_ids or self._feedback_token_ids(feedback_text),
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
                        RetryAction.STOP,
                        ControlReason.FEEDBACK_QUEUE_FULL,
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
                RetryControl(
                    completion,
                    decision.action,
                    decision.reason,
                    trained and decision.action is RetryAction.RETRY,
                )
            )
            barriers.append(
                UpdateBarrier(
                    "fttt:"
                    + completion.group_id
                    + ":"
                    + str(completion.identity.prompt_idx)
                    + ":"
                    + str(completion.identity.sample_idx),
                    completion.group_id,
                    BarrierState.READY if trained else BarrierState.BLOCKED,
                    (completion.identity.request_id,),
                    int(trained),
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
            batch,
            tuple(selected_rows),
            intent,
        )
        self._queue.append(transaction)
        self._projected_instances = projected_instances
        self._dropped_feedback = dropped_feedback
        self._next_transaction_id += 1
        self._queued_feedback += len(records)
        return bool(records)

    def begin(self, request: CompletionInput | NoHostRequest) -> FTTTHostTransaction | None:
        if self._active is not None:
            raise RuntimeError("active FTTT transaction is unresolved")
        if isinstance(request, CompletionInput):
            self.enqueue(request)
        if not self._queue:
            return None
        transaction = self._queue.popleft()
        self._queued_feedback -= len(transaction.intent.feedback_records)
        self._active = transaction
        return transaction

    def commit(
        self,
        transaction: FTTTHostTransaction,
        receipt: HostReceipt[FTTTResult],
    ) -> None:
        intent = receipt.result.intent
        if self._active is not transaction:
            raise RuntimeError("FTTT service has no active transaction")
        expected = replace(
            transaction.intent,
            adapter_version=intent.adapter_version,
        )
        if intent != expected:
            raise RuntimeError("FTTT execution rewrote the host-service intent")
        settled = dict(self._instances)
        for transition in intent.transitions:
            current = settled.get(transition.key, FTTTInstanceState())
            if current.retries != transition.previous.retries or current.stopped != transition.previous.stopped:
                raise RuntimeError("FTTT lifecycle transition is stale")
            version = intent.adapter_version if transition.trained else current.adapter_version
            settled[transition.key] = replace(
                transition.next,
                adapter_version=version,
            )
        projected = self._projected_instances if self._queue else dict(settled)
        self._instances = settled
        self._active = None
        self._projected_instances = projected

    def abort(self, transaction: FTTTHostTransaction) -> None:
        if self._active is not transaction:
            raise RuntimeError("FTTT abort does not match active transaction")
        self._active = None
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
                RetryAction.STOP,
                ControlReason.INSTANCE_STOPPED,
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
            or bool(tokens & positive)
            and not bool(tokens & negative)
        ):
            return _LifecycleDecision(
                False,
                replace(current, stopped=True),
                RetryAction.ACCEPT,
                ControlReason.POSITIVE_FEEDBACK,
            )
        retries = current.retries + 1
        stopped = self._config.max_retries > 0 and retries >= self._config.max_retries
        return _LifecycleDecision(
            True,
            FTTTInstanceState(retries, stopped, current.adapter_version),
            RetryAction.STOP if stopped else RetryAction.RETRY,
            ControlReason.RETRY_LIMIT if stopped else ControlReason.FEEDBACK_UPDATE,
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
