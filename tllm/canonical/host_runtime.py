"""Workload-independent physical facts for synchronous typed host execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import torch

from tllm.contracts.semantic_graph import (
    AnchorSymbol,
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


class HostTransactionService(
    Protocol[RequestT_contra, TransactionT, ReceiptT_contra]
):
    def begin(self, request: RequestT_contra) -> TransactionT | None: ...

    def commit(
        self, transaction: TransactionT, receipt: ReceiptT_contra
    ) -> None: ...

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
        self.execution.commit(receipt)
        return result


@dataclass(frozen=True, slots=True)
class CanonicalHostRuntimeFacts:
    device: torch.device
    plan_epoch: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.device) is not torch.device
            or self.device.type not in ("cpu", "cuda")
            or type(self.plan_epoch) is not int
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
    "HostTransactionCoordinator",
    "HostTransactionExecution",
    "HostTransactionService",
    "NO_HOST_REQUEST",
    "NoHostRequest",
]
