"""Atomic logical publication over an already-reserved CUDA PTTS write."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch

from tllm.canonical.ptts.dense_page_cuda import CudaDensePageStateProvider
from tllm.canonical.ptts.dense_page_cuda_types import (
    CudaDensePagePublication,
    CudaDensePageTransition,
)
from tllm.canonical.ptts.request_state import (
    RequestStateDirectory,
    RequestStateTransitionBatch,
)
from tllm.contracts.state_ref import StateRef


@dataclass(frozen=True, slots=True)
class CudaRequestStatePreparedPublication:
    receipt_id: int


@dataclass(frozen=True, slots=True)
class CudaRequestStatePublicationReceipt:
    """One receipt for a logical transition published by a CUDA fence."""

    operation_id: int
    physical_transaction_id: int
    before_refs: tuple[StateRef, ...]
    after_refs: tuple[StateRef, ...]
    publication: CudaDensePagePublication


@dataclass(frozen=True, slots=True)
class _PreparedPublication:
    token: CudaRequestStatePreparedPublication
    logical: RequestStateTransitionBatch
    physical: CudaDensePageTransition
    target_tensors: tuple[tuple[torch.Tensor, ...], ...]
    stream: torch.cuda.Stream


class CudaRequestStatePublicationCoordinator:
    """Commit logical latest roots only after CUDA physical publication."""

    __slots__ = (
        "_directory",
        "_lock",
        "_next_receipt_id",
        "_prepared",
        "_provider",
    )

    def __init__(
        self,
        directory: RequestStateDirectory,
        provider: CudaDensePageStateProvider,
    ) -> None:
        self._directory = directory
        self._provider = provider
        self._prepared: dict[int, _PreparedPublication] = {}
        self._next_receipt_id = 1
        self._lock = RLock()

    def prepare_full_root(
        self,
        logical: RequestStateTransitionBatch,
        physical: CudaDensePageTransition,
        target_tensors: tuple[tuple[torch.Tensor, ...], ...],
        stream: torch.cuda.Stream,
    ) -> CudaRequestStatePreparedPublication:
        """Validate and issue an opaque token before CUDA submission starts."""

        with self._lock:
            self._directory.validate_transition_publication(logical)
            self._provider.validate_full_root_transition(
                physical,
                logical.before_refs,
                logical.after_refs,
                target_tensors,
                stream,
            )
            token = CudaRequestStatePreparedPublication(self._next_receipt_id)
            self._next_receipt_id += 1
            self._prepared[token.receipt_id] = _PreparedPublication(
                token,
                logical,
                physical,
                target_tensors,
                stream,
            )
            return token

    def commit_full_root(
        self,
        token: CudaRequestStatePreparedPublication,
    ) -> CudaRequestStatePublicationReceipt:
        """Cross the fail-stop CUDA submission point, then publish logical latest."""

        with self._lock:
            prepared = self._issued(token)
            publication = self._provider.submit_full_root_transition(
                prepared.physical,
                prepared.target_tensors,
                prepared.stream,
            )
            self._directory.commit_transition(prepared.logical)
            receipt = CudaRequestStatePublicationReceipt(
                prepared.logical.operation_id,
                prepared.physical.transaction_id,
                prepared.logical.before_refs,
                prepared.logical.after_refs,
                publication,
            )
            self._prepared.pop(token.receipt_id)
            return receipt

    def abort_reservation(
        self,
        logical: RequestStateTransitionBatch,
        physical: CudaDensePageTransition,
    ) -> None:
        """Abort only a reservation that has not acquired a CUDA fence."""

        with self._lock:
            self._provider.abort_unsubmitted_transition(physical)
            self._directory.abort_transition(logical)

    def abort_prepared(
        self,
        token: CudaRequestStatePreparedPublication,
    ) -> None:
        with self._lock:
            prepared = self._issued(token)
            self._provider.abort_unsubmitted_transition(prepared.physical)
            self._directory.abort_transition(prepared.logical)
            self._prepared.pop(token.receipt_id)

    def _issued(
        self,
        token: CudaRequestStatePreparedPublication,
    ) -> _PreparedPublication:
        try:
            prepared = self._prepared[token.receipt_id]
        except KeyError as exc:
            raise RuntimeError("CUDA publication token is unknown or consumed") from exc
        if prepared.token is not token:
            raise RuntimeError("CUDA publication token differs from the issued token")
        return prepared


__all__ = [
    "CudaRequestStatePublicationCoordinator",
    "CudaRequestStatePreparedPublication",
    "CudaRequestStatePublicationReceipt",
]
