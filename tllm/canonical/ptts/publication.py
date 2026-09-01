"""Atomic visibility boundary for logical and physical request-state roots."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from tllm.canonical.ptts.dense_page import (
    DensePageStateProvider,
    DensePageTransitionBatch,
)
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory,
    RequestStateTransitionBatch,
)
from tllm.contracts.state_ref import StateRef


@dataclass(frozen=True, slots=True)
class RequestStatePublicationReceipt:
    operation_id: int
    physical_transaction_id: int
    before_refs: tuple[StateRef, ...]
    after_refs: tuple[StateRef, ...]


class RequestStatePublicationCoordinator:
    """Own the single prevalidated commit point across PTTS authorities."""

    __slots__ = ("_directory", "_lock", "_provider")

    def __init__(
        self,
        directory: RequestStateDirectory,
        provider: DensePageStateProvider,
    ) -> None:
        self._directory = directory
        self._provider = provider
        self._lock = RLock()

    def publish(
        self,
        logical: RequestStateTransitionBatch,
        physical: DensePageTransitionBatch,
    ) -> RequestStatePublicationReceipt:
        with self._lock:
            physical_before = tuple(
                transition.source for transition in physical.transitions
            )
            physical_after = tuple(
                transition.target for transition in physical.transitions
            )
            if (
                logical.before_refs != physical_before
                or logical.after_refs != physical_after
            ):
                raise ValueError("logical and physical PTTS publications differ")
            self._directory.validate_transition_publication(logical)
            self._provider.validate_transition_publication(physical)
            receipt = RequestStatePublicationReceipt(
                logical.operation_id,
                physical.transaction_id,
                logical.before_refs,
                logical.after_refs,
            )
            self._provider.publish_transition_batch(physical)
            self._directory.commit_transition(logical)
            return receipt

    def abort(
        self,
        logical: RequestStateTransitionBatch,
        physical: DensePageTransitionBatch,
    ) -> None:
        with self._lock:
            self._provider.abort_transition_batch(physical)
            self._directory.abort_transition(logical)


__all__ = [
    "RequestStatePublicationCoordinator",
    "RequestStatePublicationReceipt",
]
