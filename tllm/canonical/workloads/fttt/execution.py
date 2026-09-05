"""Transactional execution of the Rust-planned canonical FTTT graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.canonical.execution import TaskGraphExecution
from tllm.canonical.scheduler import validate_physical_plan, validate_schedule
from tllm.canonical.workloads.fttt.tasks import (
    FTTT_CUDA_LOSS_IMPLEMENTATION,
    FTTT_CUDA_OPTIMIZE_IMPLEMENTATION,
    FTTT_CUDA_PREDICT_IMPLEMENTATION,
    FTTT_CUDA_PREPARE_IMPLEMENTATION,
    FTTTEditState,
    FTTT_LOSS_IMPLEMENTATION,
    FTTTLossPending,
    FTTTLossTask,
    FTTTNotUpdated,
    FTTT_OPTIMIZE_IMPLEMENTATION,
    FTTTOptimizationPending,
    FTTTOptimizeTask,
    FTTTOptimized,
    FTTTPredicted,
    FTTT_PREDICT_IMPLEMENTATION,
    FTTTPredictionPending,
    FTTTPredictTask,
    FTTTPreparePending,
    FTTT_PREPARE_IMPLEMENTATION,
    FTTTPrepareTask,
    FTTTPublicationPending,
    FTTTPublished,
    FTTT_PUBLISH_IMPLEMENTATION,
    FTTTPublishTask,
    FTTTStepContext,
    initialize_edit_state,
)
from tllm.canonical.workloads.fttt.types import (
    FTTTClosedLoopIntent,
    FTTTConfig,
    FTTTCpuReadiness,
    FTTTCudaReadiness,
    FTTTHostTransaction,
    FTTTSpecialization,
)
from tllm.contracts.canonical_compilation import CanonicalScheduleBundle
from tllm.contracts.semantic_graph import (
    ImplementationId,
    PhysicalPlan,
)


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
    completion: "FTTTExecutionCompletion"

    def __post_init__(self) -> None:
        if type(self.receipt_id) is not int or self.receipt_id < 1:
            raise ValueError("FTTT execution receipt requires a positive ID")


@dataclass(frozen=True, slots=True)
class FTTTCpuCompletion:
    pass


@dataclass(frozen=True, slots=True)
class FTTTCudaCompletion:
    stream: torch.cuda.Stream
    event: torch.cuda.Event

    def __post_init__(self) -> None:
        if (
            type(self.stream) is not torch.cuda.Stream
            or type(self.event) is not torch.cuda.Event
        ):
            raise TypeError("FTTT CUDA completion requires an exact stream event")


FTTTExecutionCompletion = FTTTCpuCompletion | FTTTCudaCompletion


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


@dataclass(frozen=True, slots=True)
class _Failed:
    pass


_TransactionState = _Ready | _Pending | _Failed


@dataclass(frozen=True, slots=True)
class _CpuExecution:
    pass


@dataclass(frozen=True, slots=True)
class _CudaExecution:
    stream: torch.cuda.Stream


_ExecutionDevice = _CpuExecution | _CudaExecution


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
        if specialization.device.type == "cuda":
            torch.cuda.synchronize(specialization.device)
        implementations_by_task = {
            int(task.task_id): task.implementation_id
            for task in schedule_bundle.schedule_physical.tasks
        }
        if tuple(sorted(implementations_by_task)) != tuple(range(5)):
            raise ValueError("FTTT physical plan does not cover dense task IDs")
        implementation_ids: tuple[
            ImplementationId,
            ImplementationId,
            ImplementationId,
            ImplementationId,
            ImplementationId,
        ] = (
            implementations_by_task[0],
            implementations_by_task[1],
            implementations_by_task[2],
            implementations_by_task[3],
            implementations_by_task[4],
        )
        match specialization.device.type:
            case "cpu":
                expected_implementation_ids = (
                    FTTT_LOSS_IMPLEMENTATION,
                    FTTT_OPTIMIZE_IMPLEMENTATION,
                    FTTT_PREDICT_IMPLEMENTATION,
                    FTTT_PREPARE_IMPLEMENTATION,
                    FTTT_PUBLISH_IMPLEMENTATION,
                )
            case "cuda":
                expected_implementation_ids = (
                    FTTT_CUDA_LOSS_IMPLEMENTATION,
                    FTTT_CUDA_OPTIMIZE_IMPLEMENTATION,
                    FTTT_CUDA_PREDICT_IMPLEMENTATION,
                    FTTT_CUDA_PREPARE_IMPLEMENTATION,
                    FTTT_PUBLISH_IMPLEMENTATION,
                )
            case _:
                raise ValueError("FTTT specialization device is unsupported")
        if implementation_ids != expected_implementation_ids:
            raise ValueError("FTTT physical implementations differ from placement")
        self._runner = TaskGraphExecution(
            schedule_bundle.schedule_physical,
            (
                FTTTLossTask(implementation_ids[0]),
                FTTTOptimizeTask(config, implementation_ids[1]),
                FTTTPredictTask(implementation_ids[2]),
                FTTTPrepareTask(config, implementation_ids[3]),
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
            case _Failed():
                raise RuntimeError("FTTT CUDA execution is fail-stopped")
        if (
            type(transaction) is not FTTTHostTransaction
            or int(transaction.hidden_rows.shape[1])
            != self._specialization.hidden_size
            or transaction.hidden_rows.dtype is not torch.float32
            or transaction.hidden_rows.device != self._specialization.device
        ):
            raise ValueError("FTTT transaction differs from specialization")
        match transaction.readiness:
            case FTTTCpuReadiness() if self._specialization.device.type == "cpu":
                execution_device: _ExecutionDevice = _CpuExecution()
            case FTTTCudaReadiness() as readiness if (
                self._specialization.device.type == "cuda"
            ):
                stream = torch.cuda.current_stream(self._specialization.device)
                readiness.order_on(stream)
                execution_device = _CudaExecution(stream)
            case FTTTCpuReadiness() | FTTTCudaReadiness():
                raise ValueError("FTTT transaction readiness differs from placement")
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
        try:
            with torch.inference_mode(False), torch.enable_grad():
                completed = self._runner.run(context)
        except Exception:
            try:
                self._restore_snapshot(snapshot, execution_device)
            except Exception:
                self._transaction_state = _Failed()
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
                try:
                    self._restore_snapshot(snapshot, execution_device)
                except Exception:
                    self._transaction_state = _Failed()
                raise RuntimeError("FTTT signed graph did not publish all outputs")
        match execution_device:
            case _CpuExecution():
                completion: FTTTExecutionCompletion = FTTTCpuCompletion()
            case _CudaExecution(stream=stream):
                try:
                    event = torch.cuda.Event(blocking=False)
                    event.record(stream)
                except Exception:
                    try:
                        self._restore_snapshot(snapshot, execution_device)
                    except Exception:
                        self._transaction_state = _Failed()
                    raise
                try:
                    event.synchronize()
                except Exception:
                    self._transaction_state = _Failed()
                    raise
                completion = FTTTCudaCompletion(stream, event)
        result = FTTTExecutionResult(
            transaction.plan.intent.transaction_id,
            delta,
            intent,
            losses,
            update_count,
            version,
        )
        receipt = FTTTExecutionReceipt(
            self._next_receipt_id,
            result,
            completion,
        )
        self._next_receipt_id += 1
        self._transaction_state = _Pending(receipt, snapshot)
        return receipt

    def commit(self, receipt: FTTTExecutionReceipt) -> None:
        self.validate_receipt(receipt)
        self._transaction_count += 1
        self._loss_count += len(receipt.result.losses)
        self._update_count += receipt.result.update_count
        self._transaction_state = _Ready()

    def result(self, receipt: FTTTExecutionReceipt) -> FTTTExecutionResult:
        self.validate_receipt(receipt)
        return receipt.result

    def validate_receipt(self, receipt: FTTTExecutionReceipt) -> None:
        match self._transaction_state:
            case _Pending(receipt=issued) if issued is receipt:
                pass
            case _Ready() | _Pending():
                raise RuntimeError("FTTT commit receipt is not active")
            case _Failed():
                raise RuntimeError("FTTT CUDA execution is fail-stopped")
        if (
            receipt.result.state_version != self._edit_state.version
            or receipt.result.intent.adapter_version != self._edit_state.version
        ):
            raise RuntimeError("FTTT receipt differs from the execution state version")
        match receipt.completion:
            case FTTTCpuCompletion() if self._specialization.device.type == "cpu":
                pass
            case FTTTCudaCompletion(event=event) if (
                self._specialization.device.type == "cuda" and event.query()
            ):
                pass
            case FTTTCpuCompletion() | FTTTCudaCompletion():
                raise RuntimeError("FTTT execution completion is not ready")

    def abort(self, receipt: FTTTExecutionReceipt) -> None:
        match self._transaction_state:
            case _Pending(receipt=issued, snapshot=snapshot) if issued is receipt:
                pass
            case _Ready() | _Pending():
                raise RuntimeError("FTTT abort receipt is not active")
            case _Failed():
                raise RuntimeError("FTTT CUDA execution is fail-stopped")
        match receipt.completion:
            case FTTTCpuCompletion():
                execution_device: _ExecutionDevice = _CpuExecution()
            case FTTTCudaCompletion(stream=stream):
                execution_device = _CudaExecution(stream)
        try:
            self._restore_snapshot(snapshot, execution_device)
        except Exception:
            self._transaction_state = _Failed()
            raise
        self._transaction_state = _Ready()

    def _restore_snapshot(
        self,
        snapshot: _StateSnapshot,
        execution_device: _ExecutionDevice,
    ) -> None:
        match execution_device:
            case _CpuExecution():
                snapshot.restore(self._edit_state)
            case _CudaExecution(stream=stream):
                with torch.cuda.stream(stream):
                    snapshot.restore(self._edit_state)
                stream.synchronize()

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
    "FTTTExecutionCompletion",
    "FTTTExecutionReceipt",
    "FTTTExecutionResult",
    "FTTTCpuCompletion",
    "FTTTCudaCompletion",
]
