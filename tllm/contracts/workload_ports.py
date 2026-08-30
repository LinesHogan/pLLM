"""Closed value and transaction contracts shared across workload layers."""

from __future__ import annotations

from enum import Enum, StrEnum


class RuntimeValueMarker(Enum):
    SKIPPED = "skipped"


SKIPPED = RuntimeValueMarker.SKIPPED


class StagedLifecycle(StrEnum):
    DENSE_OPERAND_CAPTURED = "dense_operand.captured"
    REQUEST_FINISHED = "request.finished"


class PortTransactionStatus(StrEnum):
    PENDING = "pending"
    CONSUMED = "consumed"
    ABORTED = "aborted"


class StagedPortTransactionResolution(StrEnum):
    CANCELLED = "cancelled"
    CONSUMED = "consumed"


__all__ = [
    "PortTransactionStatus",
    "RuntimeValueMarker",
    "SKIPPED",
    "StagedLifecycle",
    "StagedPortTransactionResolution",
]
