"""Atomic composition of the FTTT host service and signed execution."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.canonical.workloads.fttt.execution import (
    FTTTExecution,
    FTTTExecutionResult,
)
from tllm.canonical.workloads.fttt.host_service import FTTTFeedbackService
from tllm.canonical.workloads.fttt.types import (
    FTTTHostTransaction,
    FTTTNoTransaction,
)


@dataclass(frozen=True, slots=True)
class FTTTNoExecution:
    pass


@dataclass(frozen=True, slots=True)
class FTTTExecuted:
    result: FTTTExecutionResult


FTTTSessionPoll = FTTTNoExecution | FTTTExecuted


class FTTTSession:
    """Resolve each host transaction exactly once across both authorities."""

    __slots__ = ("_execution", "_service")

    def __init__(
        self,
        service: FTTTFeedbackService,
        execution: FTTTExecution,
    ) -> None:
        if (
            type(service) is not FTTTFeedbackService
            or type(execution) is not FTTTExecution
            or service.config != execution.config
        ):
            raise TypeError("FTTT session requires one exact shared configuration")
        service.bind_execution(execution)
        self._service = service
        self._execution = execution

    def run_next(self) -> FTTTSessionPoll:
        transaction = self._service.poll()
        match transaction:
            case FTTTNoTransaction():
                return FTTTNoExecution()
            case FTTTHostTransaction():
                pass
        try:
            receipt = self._execution.execute(transaction)
        except Exception:
            self._service.abort(transaction.plan.intent.transaction_id)
            raise
        try:
            self._service.commit(receipt)
        except Exception:
            try:
                self._execution.abort(receipt)
            finally:
                self._service.abort(transaction.plan.intent.transaction_id)
            raise
        return FTTTExecuted(receipt.result)


__all__ = [
    "FTTTExecuted",
    "FTTTNoExecution",
    "FTTTSession",
    "FTTTSessionPoll",
]
