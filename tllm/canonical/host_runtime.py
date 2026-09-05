"""Workload-independent physical facts for synchronous typed host execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import torch

from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.canonical.execution import TaskGraphExecution, TaskGraphImplementation
from tllm.canonical.scheduler import compile_schedule, lower_physical_plan
from tllm.contracts.canonical_compilation import CanonicalScheduleBundle
from tllm.contracts.semantic_graph import (
    AnchorSymbol,
    PhysicalPlan,
    SemanticGraph,
    SchedulerTimeline,
    TimelineAnchor,
)


RequestT = TypeVar("RequestT")
TransactionT = TypeVar("TransactionT")
ReceiptT = TypeVar("ReceiptT")
ResultT = TypeVar("ResultT")
RequestT_contra = TypeVar("RequestT_contra", contravariant=True)
TransactionT_contra = TypeVar("TransactionT_contra", contravariant=True)
ReceiptT_contra = TypeVar("ReceiptT_contra", contravariant=True)
ResultT_co = TypeVar("ResultT_co", covariant=True)
ContextT = TypeVar("ContextT")
SnapshotT = TypeVar("SnapshotT")


class HostTaskState(Protocol[TransactionT_contra, ContextT, SnapshotT, ResultT_co]):
    """Workload mathematics and reversible state, without transaction ownership."""

    @property
    def version(self) -> int: ...

    def prepare(self, transaction: TransactionT_contra) -> ContextT:
        """Validate/order input readiness without mutating observable state."""
        ...

    def snapshot(self) -> SnapshotT: ...

    def restore(self, snapshot: SnapshotT) -> None: ...

    def output(self, context: ContextT) -> ResultT_co: ...


@dataclass(frozen=True, slots=True)
class HostReceipt(Generic[ResultT]):
    receipt_id: int
    result: ResultT
    completion: torch.cuda.Event | None


@dataclass(frozen=True, slots=True)
class _Pending(Generic[ResultT, SnapshotT]):
    receipt: HostReceipt[ResultT]
    snapshot: SnapshotT
    stream: torch.cuda.Stream | None


@dataclass(frozen=True, slots=True)
class _Failed:
    pass


@dataclass(frozen=True, slots=True)
class HostDiagnostics:
    transaction_count: int
    state_version: int
    fallback_count: int


class HostExecution(Generic[TransactionT, ContextT, SnapshotT, ResultT]):
    """One receipt, rollback and completion owner for signed host task graphs."""

    def __init__(
        self,
        graph: SemanticGraph,
        timeline: SchedulerTimeline,
        state: HostTaskState[TransactionT, ContextT, SnapshotT, ResultT],
        implementations: tuple[TaskGraphImplementation[ContextT], ...],
        device: torch.device,
        *,
        plan_epoch: int = 1,
    ) -> None:
        decision = compile_schedule(graph, timeline)
        physical = lower_physical_plan(graph, timeline, decision, plan_epoch=plan_epoch)
        self.schedule_bundle = CanonicalScheduleBundle(graph, timeline, decision, physical)
        self._runner = TaskGraphExecution(physical, implementations)
        self._state = state
        self._device = device
        self._initialization = (
            CudaStreamReadiness(torch.cuda.current_stream(device)) if device.type == "cuda" else None
        )
        self._active: _Pending[ResultT, SnapshotT] | _Failed | None = None
        self._next_receipt_id = 1
        self._transaction_count = 0

    @property
    def plan(self) -> PhysicalPlan:
        return self.schedule_bundle.schedule_physical

    def execute(self, transaction: TransactionT) -> HostReceipt[ResultT]:
        if self._active is not None:
            raise RuntimeError("host execution is pending or fail-stopped")
        stream = torch.cuda.current_stream(self._device) if self._device.type == "cuda" else None
        if stream is not None and self._initialization is not None:
            self._initialization.order_on(stream)
        # Readiness is ordered before the snapshot; no workload owns a second cursor.
        with torch.inference_mode(False), torch.enable_grad():
            context = self._state.prepare(transaction)
            snapshot = self._state.snapshot()
            try:
                result = self._state.output(self._runner.run(context))
                completion = None
                if stream is not None:
                    completion = torch.cuda.Event(blocking=False)
                    completion.record(stream)
                    completion.synchronize()
            except Exception:
                self._restore(snapshot, stream)
                raise
        receipt = HostReceipt(self._next_receipt_id, result, completion)
        self._next_receipt_id += 1
        self._active = _Pending(receipt, snapshot, stream)
        return receipt

    def validate_receipt(self, receipt: HostReceipt[ResultT]) -> None:
        match self._active:
            case _Pending(receipt=issued) if issued is receipt:
                pass
            case _:
                raise RuntimeError("host execution receipt is not active")

    def result(self, receipt: HostReceipt[ResultT]) -> ResultT:
        self.validate_receipt(receipt)
        return receipt.result

    def commit(self, receipt: HostReceipt[ResultT]) -> None:
        self.validate_receipt(receipt)
        self._transaction_count += 1
        self._active = None

    def abort(self, receipt: HostReceipt[ResultT]) -> None:
        self.validate_receipt(receipt)
        match self._active:
            case _Pending(snapshot=snapshot, stream=stream):
                self._restore(snapshot, stream)
                self._active = None
            case _:
                raise RuntimeError("host execution lost its active receipt")

    def _restore(self, snapshot: SnapshotT, stream: torch.cuda.Stream | None) -> None:
        try:
            if stream is None:
                self._state.restore(snapshot)
            else:
                stream.synchronize()
                with torch.cuda.stream(stream):
                    self._state.restore(snapshot)
                stream.synchronize()
        except Exception:
            self._active = _Failed()
            raise

    def diagnostics(self) -> HostDiagnostics:
        return HostDiagnostics(self._transaction_count, self._state.version, self.plan.fallback_count)


class HostTransactionService(
    Protocol[RequestT_contra, TransactionT, ReceiptT_contra]
):
    def begin(self, request: RequestT_contra) -> TransactionT | None: ...

    def commit(
        self, transaction: TransactionT, receipt: ReceiptT_contra
    ) -> None:
        """Publish atomically: an exception must leave the transaction abortable."""
        ...

    def abort(self, transaction: TransactionT) -> None: ...


class HostTransactionExecution(
    Protocol[TransactionT_contra, ReceiptT, ResultT_co]
):
    def execute(self, transaction: TransactionT_contra) -> ReceiptT: ...

    def validate_receipt(self, receipt: ReceiptT) -> None: ...

    def commit(self, receipt: ReceiptT) -> None: ...

    def abort(self, receipt: ReceiptT) -> None: ...

    def result(self, receipt: ReceiptT) -> ResultT_co: ...


@dataclass(frozen=True, slots=True)
class NoHostRequest:
    """Closed request value for a service driven only by its owned queue."""


NO_HOST_REQUEST = NoHostRequest()


class HostSettlementError(RuntimeError, Generic[ReceiptT]):
    """The service committed; settle this exact receipt without replaying input."""

    def __init__(self, receipt: ReceiptT) -> None:
        super().__init__("host publication committed; execution settlement remains pending")
        self.receipt = receipt


@dataclass(frozen=True, slots=True)
class HostTransactionCoordinator(
    Generic[RequestT, TransactionT, ReceiptT, ResultT]
):
    """Atomically settle one typed host transaction and execution receipt."""

    service: HostTransactionService[RequestT, TransactionT, ReceiptT]
    execution: HostTransactionExecution[TransactionT, ReceiptT, ResultT]

    def run(self, request: RequestT) -> ResultT | None:
        transaction = self.service.begin(request)
        if transaction is None:
            return None
        try:
            receipt = self.execution.execute(transaction)
        except Exception:
            self.service.abort(transaction)
            raise
        try:
            self.execution.validate_receipt(receipt)
            result = self.execution.result(receipt)
            self.service.commit(transaction, receipt)
        except Exception:
            try:
                self.execution.abort(receipt)
            finally:
                self.service.abort(transaction)
            raise
        try:
            self.execution.commit(receipt)
        except Exception as exc:
            raise HostSettlementError(receipt) from exc
        return result


@dataclass(frozen=True, slots=True)
class CanonicalHostRuntimeFacts:
    device: torch.device
    plan_epoch: int = 1

    def __post_init__(self) -> None:
        if (
            self.device.type not in ("cpu", "cuda")
            or self.plan_epoch < 1
        ):
            raise ValueError("canonical host runtime facts are not closed")

    def timeline(
        self,
        extra_capabilities: tuple[str, ...] = (),
    ) -> SchedulerTimeline:
        compute_capability = (
            "compute/cpu@1"
            if self.device.type == "cpu"
            else "compute/cuda@1"
        )
        return SchedulerTimeline(
            anchors=(
                TimelineAnchor(AnchorSymbol("host.input.ready"), 0),
                TimelineAnchor(AnchorSymbol("host.compute.end"), 1),
                TimelineAnchor(AnchorSymbol("host.output.publish"), 2),
                TimelineAnchor(AnchorSymbol("host.next"), 3),
            ),
            capabilities=tuple(
                sorted(
                    (
                        compute_capability,
                        "host/service@1",
                        *extra_capabilities,
                    )
                )
            ),
        )


__all__ = [
    "CanonicalHostRuntimeFacts",
    "HostExecution",
    "HostReceipt",
    "HostTransactionCoordinator",
    "HostTransactionExecution",
    "HostTransactionService",
    "NO_HOST_REQUEST",
    "NoHostRequest",
]
