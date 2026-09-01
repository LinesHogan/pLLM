"""Transactional execution of the Rust-planned canonical FTTT graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.execution import TaskGraphExecution
from tllm.canonical.scheduler import validate_physical_plan, validate_schedule
from tllm.canonical.workloads.fttt.tasks import (
    FTTTEditState,
    FTTTLossPending,
    FTTTLossTask,
    FTTTNotUpdated,
    FTTTOptimizationPending,
    FTTTOptimizeTask,
    FTTTOptimized,
    FTTTPredicted,
    FTTTPredictionPending,
    FTTTPredictTask,
    FTTTPreparePending,
    FTTTPrepareTask,
    FTTTPublicationPending,
    FTTTPublished,
    FTTTPublishTask,
    FTTTStepContext,
    initialize_edit_state,
)
from tllm.canonical.workloads.fttt.types import (
    FTTTClosedLoopIntent,
    FTTTConfig,
    FTTTHostTransaction,
    FTTTSpecialization,
)
from tllm.contracts.canonical_compilation import CanonicalScheduleBundle
from tllm.contracts.semantic_graph import AnchorId, PhysicalPlan


@dataclass(frozen=True, slots=True)
class FTTTExecutionResult:
    transaction_id: int
    edit_delta: torch.Tensor
    intent: FTTTClosedLoopIntent
    losses: tuple[torch.Tensor, ...]
    update_count: int
    state_version: int


@dataclass(frozen=True, slots=True)
class FTTTExecutionReceipt:
    receipt_id: int
    result: FTTTExecutionResult

    def __post_init__(self) -> None:
        if type(self.receipt_id) is not int or self.receipt_id < 1:
            raise ValueError("FTTT execution receipt requires a positive ID")


@dataclass(frozen=True, slots=True)
class FTTTDiagnostics:
    transaction_count: int
    loss_count: int
    update_count: int
    state_version: int
    fallback_count: int


@dataclass(frozen=True, slots=True)
class _StateSnapshot:
    down: torch.Tensor
    up: torch.Tensor
    down_exp_avg: torch.Tensor
    down_exp_avg_sq: torch.Tensor
    up_exp_avg: torch.Tensor
    up_exp_avg_sq: torch.Tensor
    step: int
    version: int

    @classmethod
    def capture(cls, state: FTTTEditState) -> "_StateSnapshot":
        return cls(
            state.down.detach().clone(),
            state.up.detach().clone(),
            state.down_exp_avg.detach().clone(),
            state.down_exp_avg_sq.detach().clone(),
            state.up_exp_avg.detach().clone(),
            state.up_exp_avg_sq.detach().clone(),
            state.step,
            state.version,
        )

    def restore(self, state: FTTTEditState) -> None:
        with torch.no_grad():
            state.down.copy_(self.down)
            state.up.copy_(self.up)
            state.down_exp_avg.copy_(self.down_exp_avg)
            state.down_exp_avg_sq.copy_(self.down_exp_avg_sq)
            state.up_exp_avg.copy_(self.up_exp_avg)
            state.up_exp_avg_sq.copy_(self.up_exp_avg_sq)
        state.step = self.step
        state.version = self.version


@dataclass(frozen=True, slots=True)
class _Ready:
    pass


@dataclass(frozen=True, slots=True)
class _Pending:
    receipt: FTTTExecutionReceipt
    snapshot: _StateSnapshot


_TransactionState = _Ready | _Pending


class FTTTExecution:
    """Instance-owned edit state advanced only through signed physical tasks."""

    __slots__ = (
        "_config",
        "_edit_state",
        "_loss_count",
        "_next_receipt_id",
        "_runner",
        "_schedule_bundle",
        "_specialization",
        "_transaction_count",
        "_transaction_state",
        "_update_count",
    )

    def __init__(
        self,
        config: FTTTConfig,
        specialization: FTTTSpecialization,
        schedule_bundle: CanonicalScheduleBundle,
    ) -> None:
        if (
            type(config) is not FTTTConfig
            or type(specialization) is not FTTTSpecialization
            or type(schedule_bundle) is not CanonicalScheduleBundle
            or config.rank > specialization.hidden_size
            or schedule_bundle.semantic_graph.execution_parameters_digest
            != specialization.execution_parameters_digest(config)
        ):
            raise TypeError("FTTT execution requires exact compiled artifacts")
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
        self._config = config
        self._specialization = specialization
        self._schedule_bundle = schedule_bundle
        self._edit_state = initialize_edit_state(config, specialization)
        self._runner = TaskGraphExecution(
            schedule_bundle.schedule_physical,
            (
                FTTTLossTask(),
                FTTTOptimizeTask(config),
                FTTTPredictTask(),
                FTTTPrepareTask(config),
                FTTTPublishTask(),
            ),
        )
        self._transaction_state: _TransactionState = _Ready()
        self._next_receipt_id = 1
        self._transaction_count = 0
        self._loss_count = 0
        self._update_count = 0

    @property
    def plan(self) -> PhysicalPlan:
        return self._schedule_bundle.schedule_physical

    @property
    def config(self) -> FTTTConfig:
        return self._config

    @property
    def specialization(self) -> FTTTSpecialization:
        return self._specialization

    @property
    def execution_parameters_digest(self) -> str:
        return self._specialization.execution_parameters_digest(self._config)

    def execute(self, transaction: FTTTHostTransaction) -> FTTTExecutionReceipt:
        match self._transaction_state:
            case _Ready():
                pass
            case _Pending():
                raise RuntimeError("prior FTTT execution receipt is unresolved")
        if (
            type(transaction) is not FTTTHostTransaction
            or int(transaction.hidden_rows.shape[1])
            != self._specialization.hidden_size
            or transaction.hidden_rows.dtype is not torch.float32
            or transaction.hidden_rows.device != self._specialization.device
        ):
            raise ValueError("FTTT transaction differs from specialization")
        snapshot = _StateSnapshot.capture(self._edit_state)
        context = FTTTStepContext(
            transaction,
            self._edit_state,
            FTTTPreparePending(),
            FTTTLossPending(),
            FTTTOptimizationPending(),
            FTTTPredictionPending(),
            FTTTPublicationPending(),
        )
        invocation = self._runner.start(context)
        try:
            anchor_ids = tuple(
                dict.fromkeys(
                    AnchorId(int(task.anchor_id)) for task in self.plan.tasks
                )
            )
            for anchor_id in anchor_ids:
                self._runner.execute_anchor(invocation, anchor_id)
            completed = self._runner.finish(invocation)
        except Exception:
            snapshot.restore(self._edit_state)
            raise
        match completed.prediction, completed.publication, completed.optimization:
            case (
                FTTTPredicted(delta=delta),
                FTTTPublished(intent=intent),
                FTTTNotUpdated(version=version, update_count=update_count),
            ):
                losses: tuple[torch.Tensor, ...] = ()
            case (
                FTTTPredicted(delta=delta),
                FTTTPublished(intent=intent),
                FTTTOptimized(
                    loss=loss,
                    version=version,
                    update_count=update_count,
                ),
            ):
                losses = (loss,)
            case _:
                snapshot.restore(self._edit_state)
                raise RuntimeError("FTTT signed graph did not publish all outputs")
        result = FTTTExecutionResult(
            transaction.plan.intent.transaction_id,
            delta,
            intent,
            losses,
            update_count,
            version,
        )
        receipt = FTTTExecutionReceipt(self._next_receipt_id, result)
        self._next_receipt_id += 1
        self._transaction_state = _Pending(receipt, snapshot)
        return receipt

    def commit(self, receipt: FTTTExecutionReceipt) -> None:
        self.validate_receipt(receipt)
        self._transaction_count += 1
        self._loss_count += len(receipt.result.losses)
        self._update_count += receipt.result.update_count
        self._transaction_state = _Ready()

    def validate_receipt(self, receipt: FTTTExecutionReceipt) -> None:
        match self._transaction_state:
            case _Pending(receipt=issued) if issued is receipt:
                pass
            case _Ready() | _Pending():
                raise RuntimeError("FTTT commit receipt is not active")
        if (
            receipt.result.state_version != self._edit_state.version
            or receipt.result.intent.adapter_version != self._edit_state.version
        ):
            raise RuntimeError("FTTT receipt differs from the execution state version")

    def abort(self, receipt: FTTTExecutionReceipt) -> None:
        match self._transaction_state:
            case _Pending(receipt=issued, snapshot=snapshot) if issued is receipt:
                pass
            case _Ready() | _Pending():
                raise RuntimeError("FTTT abort receipt is not active")
        snapshot.restore(self._edit_state)
        self._transaction_state = _Ready()

    def parameter_snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._edit_state.down.detach().clone(),
            self._edit_state.up.detach().clone(),
        )

    def diagnostics(self) -> FTTTDiagnostics:
        return FTTTDiagnostics(
            self._transaction_count,
            self._loss_count,
            self._update_count,
            self._edit_state.version,
            self.plan.fallback_count,
        )


__all__ = [
    "FTTTDiagnostics",
    "FTTTExecution",
    "FTTTExecutionReceipt",
    "FTTTExecutionResult",
]
