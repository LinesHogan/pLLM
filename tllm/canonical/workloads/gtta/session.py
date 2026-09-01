"""Atomic composition of the GTTA host queue and signed execution."""

from __future__ import annotations

from tllm.canonical.workloads.gtta.environment import GTTAEnvironmentService
from tllm.canonical.workloads.gtta.execution import (
    GTTAExecution,
    GTTAExecutionResult,
)
from tllm.canonical.workloads.gtta.types import GTTAQuery


class GTTASession:
    """Settle one environment transaction across host and execution state."""

    __slots__ = ("_environment", "_execution")

    def __init__(
        self,
        environment: GTTAEnvironmentService,
        execution: GTTAExecution,
    ) -> None:
        if environment.config != execution.config:
            raise ValueError("GTTA session requires one shared configuration")
        self._environment = environment
        self._execution = execution

    def query(self, query: GTTAQuery, limit: int) -> GTTAExecutionResult:
        transaction = self._environment.begin(limit)
        try:
            receipt = self._execution.execute(transaction, query)
        except Exception:
            self._environment.abort(transaction.transaction_id)
            raise
        try:
            self._execution.validate_receipt(receipt)
            self._environment.commit(transaction.transaction_id)
        except Exception:
            self._execution.abort(receipt)
            self._environment.abort(transaction.transaction_id)
            raise
        self._execution.commit(receipt)
        return receipt.result


__all__ = ["GTTASession"]
