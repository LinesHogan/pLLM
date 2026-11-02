"""FTTT's judge, bounded feedback queue and retry lifecycle; FTTTWorkload trains the edit between begin and commit."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, replace
from hashlib import blake2b

from pllm.host import (
    BarrierState,
    CompletionInput,
    ControlReason,
    RetryAction,
    RetryControl,
    RewardOutcome,
    UpdateBarrier,
)
from pllm.workloads.fttt.judge import FTTTJudge
from pllm.workloads.fttt.types import (
    FTTTConfig,
    FTTTFeedbackRecord,
    FTTTLifecycleTransition,
    FTTTQueuedFeedback,
    FTTTRequestLifecycle,
)


@dataclass(frozen=True, slots=True)
class _LifecycleDecision:
    adaptable: bool
    next_state: FTTTRequestLifecycle
    action: RetryAction
    reason: ControlReason


class FTTTFeedbackQueue:
    """Own judge I/O, the bounded queue and the request lifecycles only."""

    __slots__ = (
        "_config",
        "_dropped_feedback",
        "_lifecycles",
        "_judge",
        "_projected_lifecycles",
        "_queue",
        "_queued_feedback",
    )

    def __init__(self, config: FTTTConfig, judge: FTTTJudge) -> None:
        self._config = config
        self._judge = judge
        self._lifecycles: dict[tuple[str, int], FTTTRequestLifecycle] = {}
        self._projected_lifecycles: dict[tuple[str, int], FTTTRequestLifecycle] = {}
        self._queue: deque[FTTTQueuedFeedback] = deque()
        self._queued_feedback = 0
        self._dropped_feedback = 0

    @property
    def queued_feedback(self) -> int:
        return self._queued_feedback

    @property
    def dropped_feedback(self) -> int:
        return self._dropped_feedback

    @property
    def config(self) -> FTTTConfig:
        return self._config

    def lifecycle(self, group_id: str, prompt_index: int) -> FTTTRequestLifecycle:
        return self._lifecycles.get((group_id, prompt_index), FTTTRequestLifecycle())

    def enqueue(self, batch: CompletionInput) -> bool:
        outcomes: list[RewardOutcome] = []
        retry_controls: list[RetryControl] = []
        barriers: list[UpdateBarrier] = []
        records: list[FTTTFeedbackRecord] = []
        transitions: list[FTTTLifecycleTransition] = []
        selected_rows: list[int] = []
        available = self._config.max_queue_size - self._queued_feedback
        projected_lifecycles = dict(self._projected_lifecycles)
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
            key = (completion.group_id, completion.identity.prompt_index)
            previous = projected_lifecycles.get(
                key,
                self._lifecycles.get(key, FTTTRequestLifecycle()),
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
                    decision.next_state,
                    trained,
                )
            )
            projected_lifecycles[key] = decision.next_state
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
                    + str(completion.identity.prompt_index)
                    + ":"
                    + str(completion.identity.sample_index),
                    completion.group_id,
                    BarrierState.READY if trained else BarrierState.BLOCKED,
                    (completion.identity.request_id,),
                    int(trained),
                    decision.reason,
                )
            )
        self._queue.append(
            FTTTQueuedFeedback(
                batch,
                tuple(selected_rows),
                tuple(outcomes),
                tuple(retry_controls),
                tuple(barriers),
                tuple(records),
                tuple(transitions),
            )
        )
        self._projected_lifecycles = projected_lifecycles
        self._dropped_feedback = dropped_feedback
        self._queued_feedback += len(records)
        return bool(records)

    def begin(self) -> FTTTQueuedFeedback | None:
        """Take the oldest queued feedback for an edit step; None when the queue is empty."""
        if not self._queue:
            return None
        queued = self._queue.popleft()
        self._queued_feedback -= len(queued.feedback_records)
        return queued

    def commit(self, queued: FTTTQueuedFeedback, version: int) -> None:
        """Apply each completion's retry decision with the edit version its step published."""
        committed = dict(self._lifecycles)
        for transition in queued.transitions:
            current = committed.get(transition.key, FTTTRequestLifecycle())
            committed[transition.key] = replace(
                transition.next,
                adapter_version=version if transition.trained else current.adapter_version,
            )
        projected = self._projected_lifecycles if self._queue else dict(committed)
        self._lifecycles = committed
        self._projected_lifecycles = projected

    def abort(self) -> None:
        """A failed edit step drops the whole queue; lifecycles stay as last committed."""
        self._queue.clear()
        self._queued_feedback = 0
        self._projected_lifecycles = dict(self._lifecycles)

    def _decision(
        self,
        record: FTTTFeedbackRecord,
        current: FTTTRequestLifecycle,
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
            FTTTRequestLifecycle(retries, stopped, current.adapter_version),
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
    "FTTTFeedbackQueue",
]
