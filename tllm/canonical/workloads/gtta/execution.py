"""Transactional execution of the Rust-planned canonical GTTA graph."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.canonical.execution import TaskGraphExecution
from tllm.canonical.scheduler import validate_physical_plan, validate_schedule
from tllm.canonical.workloads.gtta.state import (
    GTTADynamicsSnapshot,
    GTTADynamicsState,
)
from tllm.canonical.workloads.gtta.tasks import (
    GTTAPublishPending,
    GTTAPublished,
    GTTAPublishTask,
    GTTAQueryPending,
    GTTAQueryTask,
    GTTAStepContext,
    GTTAUpdatePending,
    GTTAUpdateTask,
    GTTA_PUBLISH_IMPLEMENTATION,
    GTTA_QUERY_IMPLEMENTATION,
    GTTA_UPDATE_IMPLEMENTATION,
)
from tllm.canonical.workloads.gtta.types import (
    GTTAConfig,
    GTTAHostTransaction,
    GTTAQuery,
    GTTAQueryResult,
    GTTAUpdateReceipt,
)
from tllm.contracts.canonical_compilation import CanonicalScheduleBundle
from tllm.contracts.semantic_graph import AnchorId, ImplementationId, PhysicalPlan


@dataclass(frozen=True, slots=True)
class GTTAExecutionResult:
    transaction_id: int
    update: GTTAUpdateReceipt
    query: GTTAQueryResult


@dataclass(frozen=True, slots=True)
class GTTAExecutionReceipt:
    receipt_id: int
    result: GTTAExecutionResult


@dataclass(frozen=True, slots=True)
class GTTADiagnostics:
    transaction_count: int
    processed_trajectory_count: int
    state_version: int
    fallback_count: int


@dataclass(frozen=True, slots=True)
class _Ready:
    pass


@dataclass(frozen=True, slots=True)
class _Pending:
    receipt: GTTAExecutionReceipt
    snapshot: GTTADynamicsSnapshot


_TransactionState = _Ready | _Pending


class GTTAExecution:
    """Own instance dynamics and advance them only through signed tasks."""

    __slots__ = (
        "_config",
        "_dynamics",
        "_next_receipt_id",
        "_processed_trajectory_count",
        "_runner",
        "_schedule_bundle",
        "_transaction_count",
        "_transaction_state",
    )

    def __init__(
        self,
        config: GTTAConfig,
        schedule_bundle: CanonicalScheduleBundle,
    ) -> None:
        if (
            schedule_bundle.semantic_graph.execution_parameters_digest
            != config.execution_parameters_digest()
        ):
            raise ValueError("GTTA execution parameters differ from the signed graph")
        validate_schedule(
            schedule_bundle.semantic_graph,
            schedule_bundle.timeline,
            schedule_bundle.schedule_decision,
        )
        validate_physical_plan(
            schedule_bundle.semantic_graph,
            schedule_bundle.timeline,
            schedule_bundle.schedule_decision,
            schedule_bundle.schedule_physical,
        )
        implementations_by_task = {
            int(task.task_id): task.implementation_id
            for task in schedule_bundle.schedule_physical.tasks
        }
        if tuple(sorted(implementations_by_task)) != (0, 1, 2):
            raise ValueError("GTTA physical plan does not cover dense task IDs")
        implementation_ids: tuple[
            ImplementationId,
            ImplementationId,
            ImplementationId,
        ] = (
            implementations_by_task[0],
            implementations_by_task[1],
            implementations_by_task[2],
        )
        if implementation_ids != (
            GTTA_PUBLISH_IMPLEMENTATION,
            GTTA_QUERY_IMPLEMENTATION,
            GTTA_UPDATE_IMPLEMENTATION,
        ):
            raise ValueError("GTTA physical implementations differ from placement")
        self._config = config
        self._schedule_bundle = schedule_bundle
        self._dynamics = GTTADynamicsState()
        self._runner = TaskGraphExecution(
            schedule_bundle.schedule_physical,
            (
                GTTAPublishTask(),
                GTTAQueryTask(),
                GTTAUpdateTask(config),
            ),
        )
        self._transaction_state: _TransactionState = _Ready()
        self._next_receipt_id = 1
        self._transaction_count = 0
        self._processed_trajectory_count = 0

    @property
    def plan(self) -> PhysicalPlan:
        return self._schedule_bundle.schedule_physical

    @property
    def config(self) -> GTTAConfig:
        return self._config

    @property
    def execution_parameters_digest(self) -> str:
        return self._config.execution_parameters_digest()

    def execute(
        self,
        transaction: GTTAHostTransaction,
        query: GTTAQuery,
    ) -> GTTAExecutionReceipt:
        match self._transaction_state:
            case _Ready():
                pass
            case _Pending():
                raise RuntimeError("prior GTTA execution receipt is unresolved")
        snapshot = self._dynamics.snapshot()
        context = GTTAStepContext(
            transaction,
            query,
            self._dynamics,
            GTTAUpdatePending(),
            GTTAQueryPending(),
            GTTAPublishPending(),
        )
        invocation = self._runner.start(context)
        try:
            anchor_ids = tuple(
                dict.fromkeys(AnchorId(int(task.anchor_id)) for task in self.plan.tasks)
            )
            for anchor_id in anchor_ids:
                self._runner.execute_anchor(invocation, anchor_id)
            completed = self._runner.finish(invocation)
        except Exception:
            self._dynamics.restore(snapshot)
            raise
        match completed.publication:
            case GTTAPublished(update=update, query=query_result):
                pass
            case GTTAPublishPending():
                self._dynamics.restore(snapshot)
                raise RuntimeError("GTTA signed graph did not publish its outputs")
        result = GTTAExecutionResult(
            transaction.transaction_id,
            update,
            query_result,
        )
        receipt = GTTAExecutionReceipt(self._next_receipt_id, result)
        self._next_receipt_id += 1
        self._transaction_state = _Pending(receipt, snapshot)
        return receipt

    def validate_receipt(self, receipt: GTTAExecutionReceipt) -> None:
        match self._transaction_state:
            case _Pending(receipt=issued) if issued is receipt:
                pass
            case _Ready() | _Pending():
                raise RuntimeError("GTTA execution receipt is not active")
        if receipt.result.update.state_version != self._dynamics.version:
            raise RuntimeError("GTTA receipt differs from instance state")

    def commit(self, receipt: GTTAExecutionReceipt) -> None:
        self.validate_receipt(receipt)
        self._transaction_count += 1
        self._processed_trajectory_count += receipt.result.update.processed
        self._transaction_state = _Ready()

    def abort(self, receipt: GTTAExecutionReceipt) -> None:
        match self._transaction_state:
            case _Pending(receipt=issued, snapshot=snapshot) if issued is receipt:
                pass
            case _Ready() | _Pending():
                raise RuntimeError("GTTA abort receipt is not active")
        self._dynamics.restore(snapshot)
        self._transaction_state = _Ready()

    def diagnostics(self) -> GTTADiagnostics:
        return GTTADiagnostics(
            self._transaction_count,
            self._processed_trajectory_count,
            self._dynamics.version,
            self.plan.fallback_count,
        )


__all__ = [
    "GTTADiagnostics",
    "GTTAExecution",
    "GTTAExecutionReceipt",
    "GTTAExecutionResult",
]
