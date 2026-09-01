"""Signed eager CUDA dense-page provider with event-gated reclamation."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from weakref import WeakValueDictionary

import torch

from tllm.canonical.ptts.dense_page_adamw import (
    DensePageAdamWConfig,
    DensePageAdamWFieldBinding,
)
from tllm.canonical.ptts.dense_page_cuda_ledger import (
    CudaDensePageAdmissionBatch,
    CudaDensePageFence,
    CudaDensePageLedger,
    CudaDensePageTransitionBatch,
)
from tllm.canonical.ptts.dense_page_cuda_types import (
    CudaDensePageAdmission,
    CudaDensePageGradientBatch,
    CudaDensePageInitialRoot,
    CudaDensePagePreparedAdmission,
    CudaDensePagePublication,
    CudaDensePageRelease,
    CudaDensePageTransition,
    CudaDensePageVersionPin,
)
from tllm.canonical.ptts.dense_page_cuda_pools import CudaDensePagePools
from tllm.canonical.ptts.dense_page_validation import (
    validate_dense_page_provider,
)
from tllm.contracts.state_ref import StateRef
from tllm.contracts.state_representation import (
    StateDeviceKind,
    StatePhysicalPlan,
    StateProviderPlan,
    StateRepresentationDecision,
    StateRepresentationRequest,
)
from tllm.contracts.tensor_layout import (
    LayoutFieldId,
    TensorLayoutDecision,
    TensorLayoutRequest,
)


@dataclass(frozen=True, slots=True)
class _PreparedAdmissionState:
    token: CudaDensePagePreparedAdmission
    batch: CudaDensePageAdmissionBatch
    admissions: tuple[CudaDensePageAdmission, ...]
    stream: torch.cuda.Stream


@dataclass(frozen=True, slots=True)
class _PreparedTransitionState:
    token: CudaDensePageTransition
    batch: CudaDensePageTransitionBatch
    stream: torch.cuda.Stream


@dataclass(frozen=True, slots=True)
class _BoundGradientState:
    token: CudaDensePageGradientBatch
    values: tuple[torch.Tensor, ...]
    stream: torch.cuda.Stream


class CudaDensePageStateProvider:
    """CUDA pools and optimizer math behind Rust-signed physical facts."""

    __slots__ = (
        "_resource",
        "_selected",
        "_pools",
        "_ledger",
        "_next_completion_fence_id",
        "_next_release_id",
        "_release_receipts",
        "_prepared_admissions",
        "_prepared_transitions",
        "_gradient_batches",
        "_next_gradient_batch_id",
        "_poisoned",
        "_lock",
    )

    def __init__(
        self,
        *,
        tensor_layout_request: TensorLayoutRequest,
        tensor_layout_decision: TensorLayoutDecision,
        representation_request: StateRepresentationRequest,
        representation_decision: StateRepresentationDecision,
        physical_plan: StatePhysicalPlan,
        local_candidates: tuple[StateProviderPlan, ...],
        resource: str,
        device_ordinal: int,
    ) -> None:
        if (
            type(device_ordinal) is not int
            or device_ordinal < 0
            or not torch.cuda.is_available()
            or device_ordinal >= torch.cuda.device_count()
        ):
            raise RuntimeError("CUDA dense-page device is unavailable")
        signed = validate_dense_page_provider(
            tensor_layout_request=tensor_layout_request,
            tensor_layout_decision=tensor_layout_decision,
            representation_request=representation_request,
            representation_decision=representation_decision,
            physical_plan=physical_plan,
            local_candidates=local_candidates,
            resource=resource,
            device_kind=StateDeviceKind.CUDA,
            device_ordinal=device_ordinal,
        )
        if signed.selected_plan.capture_stable:
            raise RuntimeError(
                "eager CUDA dense-page v1 must not claim graph capture support"
            )
        device = torch.device("cuda", device_ordinal)
        self._resource = resource
        self._selected = signed.selected_plan
        self._pools = CudaDensePagePools(signed, device)
        self._ledger = CudaDensePageLedger(
            layout_id=signed.selected_plan.layout_id,
            device_ordinal=device_ordinal,
            max_admitted_lineages=signed.selected_plan.max_admitted_lineages,
            max_live_versions_per_lineage=(
                signed.selected_plan.max_live_versions_per_lineage
            ),
        )
        self._next_completion_fence_id = 1
        self._next_release_id = 1
        self._release_receipts: WeakValueDictionary[
            int,
            CudaDensePageRelease,
        ] = WeakValueDictionary()
        self._prepared_admissions: dict[int, _PreparedAdmissionState] = {}
        self._prepared_transitions: dict[int, _PreparedTransitionState] = {}
        self._gradient_batches: dict[int, _BoundGradientState] = {}
        self._next_gradient_batch_id = 1
        self._poisoned = False
        self._lock = RLock()

    @property
    def resource(self) -> str:
        return self._resource

    @property
    def device(self) -> torch.device:
        return self._pools.device

    @property
    def selected_plan(self) -> StateProviderPlan:
        return self._selected

    @property
    def pool_bytes(self) -> int:
        return self._pools.pool_bytes

    @property
    def free_pages(self) -> int:
        with self._lock:
            self._require_live()
            return self._ledger.free_pages

    @property
    def pending_pin_releases(self) -> int:
        with self._lock:
            self._require_live()
            return self._ledger.pending_pin_releases

    @property
    def pool_addresses(self) -> tuple[int, ...]:
        return self._pools.addresses

    def require_stable(self) -> None:
        with self._lock:
            self._require_live()
            self._pools.require_stable()

    def new_zero_root(
        self,
        stream: torch.cuda.Stream,
    ) -> CudaDensePageInitialRoot:
        with self._lock:
            self._require_stream(stream)
            return self._pools.new_zero_root(stream)

    def prepare_admission_batch(
        self,
        admissions: tuple[CudaDensePageAdmission, ...],
    ) -> CudaDensePagePreparedAdmission:
        with self._lock:
            return self._prepare_admission_batch(admissions)

    def _prepare_admission_batch(
        self,
        admissions: tuple[CudaDensePageAdmission, ...],
    ) -> CudaDensePagePreparedAdmission:
        if type(admissions) is not tuple or not admissions:
            raise ValueError("CUDA dense-page admission requires roots")
        if any(
            type(admission) is not CudaDensePageAdmission
            or type(admission.initial) is not CudaDensePageInitialRoot
            for admission in admissions
        ):
            raise TypeError("CUDA admission requires typed initial roots")
        initialization_stream = admissions[0].initial.stream
        self._require_stream(initialization_stream)
        for admission in admissions:
            self._require_stream(admission.initial.stream)
            if (
                admission.initial.stream.cuda_stream
                != initialization_stream.cuda_stream
            ):
                raise RuntimeError("CUDA admission initial roots use multiple streams")
            self._pools.require_initial_root(admission.initial)
        batch = self._ledger.prepare_admission_batch(
            tuple(admission.ref for admission in admissions)
        )
        token = CudaDensePagePreparedAdmission(batch.transaction_id)
        self._prepared_admissions[token.transaction_id] = (
            _PreparedAdmissionState(
                token,
                batch,
                admissions,
                initialization_stream,
            )
        )
        return token

    def submit_admission_batch(
        self,
        prepared: CudaDensePagePreparedAdmission,
        stream: torch.cuda.Stream,
    ) -> CudaDensePagePublication:
        with self._lock:
            return self._submit_admission_batch(prepared, stream)

    def _submit_admission_batch(
        self,
        prepared: CudaDensePagePreparedAdmission,
        stream: torch.cuda.Stream,
    ) -> CudaDensePagePublication:
        self._require_stream(stream)
        state = self._prepared_admission(prepared)
        if stream.cuda_stream != state.stream.cuda_stream:
            raise RuntimeError(
                "CUDA admission submission uses another initialization stream"
            )
        if tuple(
            admission.ref for admission in state.admissions
        ) != state.batch.targets:
            raise RuntimeError(
                "CUDA admission payload differs from its reserved roots"
            )
        for admission in state.admissions:
            self._pools.require_initial_root(admission.initial)
        self._ledger.mark_submitted(state.batch)
        try:
            self._pools.submit_admission(
                state.admissions,
                state.batch.target_page_ids,
                stream,
            )
            readiness = self._record_readiness(stream)
            roots = self._ledger.publish_batch(state.batch, readiness)
            self._prepared_admissions.pop(prepared.transaction_id)
            return CudaDensePagePublication(tuple(root.ref for root in roots))
        except Exception:
            self._poison()
            raise

    def abort_unsubmitted_admission(
        self,
        prepared: CudaDensePagePreparedAdmission,
    ) -> None:
        with self._lock:
            self._abort_unsubmitted_admission(prepared)

    def _abort_unsubmitted_admission(
        self,
        prepared: CudaDensePagePreparedAdmission,
    ) -> None:
        state = self._prepared_admission(prepared)
        self._ledger.abort_unsubmitted_batch(state.batch)
        self._prepared_admissions.pop(prepared.transaction_id)

    def prepare_transition_batch(
        self,
        sources: tuple[StateRef, ...],
        targets: tuple[StateRef, ...],
        stream: torch.cuda.Stream,
    ) -> CudaDensePageTransition:
        with self._lock:
            return self._prepare_transition_batch(sources, targets, stream)

    def _prepare_transition_batch(
        self,
        sources: tuple[StateRef, ...],
        targets: tuple[StateRef, ...],
        stream: torch.cuda.Stream,
    ) -> CudaDensePageTransition:
        if type(sources) is not tuple or type(targets) is not tuple:
            raise TypeError("CUDA transition roots must be exact tuples")
        batch = self._ledger.prepare_transition_batch(sources, targets, stream)
        token = CudaDensePageTransition(batch.transaction_id)
        self._prepared_transitions[token.transaction_id] = (
            _PreparedTransitionState(token, batch, stream)
        )
        return token

    def submit_adamw_transition(
        self,
        transition: CudaDensePageTransition,
        fields: DensePageAdamWFieldBinding,
        config: DensePageAdamWConfig,
        gradients: CudaDensePageGradientBatch,
        stream: torch.cuda.Stream,
    ) -> CudaDensePagePublication:
        with self._lock:
            return self._submit_adamw_transition(
                transition,
                fields,
                config,
                gradients,
                stream,
            )

    def _submit_adamw_transition(
        self,
        transition: CudaDensePageTransition,
        fields: DensePageAdamWFieldBinding,
        config: DensePageAdamWConfig,
        gradients: CudaDensePageGradientBatch,
        stream: torch.cuda.Stream,
    ) -> CudaDensePagePublication:
        self._require_stream(stream)
        state = self._prepared_transition(transition)
        if stream.cuda_stream != state.stream.cuda_stream:
            raise RuntimeError(
                "CUDA transition submission uses another prepared stream"
            )
        bound_gradients = self._bound_gradients(gradients)
        if stream.cuda_stream != bound_gradients.stream.cuda_stream:
            raise RuntimeError(
                "CUDA gradients belong to another producer stream"
            )
        if (
            type(fields) is not DensePageAdamWFieldBinding
            or type(config) is not DensePageAdamWConfig
        ):
            raise TypeError("CUDA AdamW requires exact binding and config types")
        self._pools.require_adamw(
            fields,
            bound_gradients.values,
            len(state.batch.targets),
        )
        self._ledger.mark_submitted(state.batch)
        try:
            self._pools.submit_adamw(
                source_page_ids=state.batch.source_page_ids,
                target_page_ids=state.batch.target_page_ids,
                fields=fields,
                config=config,
                gradients=bound_gradients.values,
                stream=stream,
            )
            readiness = self._record_readiness(stream)
            roots = self._ledger.publish_batch(state.batch, readiness)
            self._prepared_transitions.pop(transition.transaction_id)
            self._gradient_batches.pop(gradients.batch_id)
            return CudaDensePagePublication(tuple(root.ref for root in roots))
        except Exception:
            self._poison()
            raise

    def validate_full_root_transition(
        self,
        transition: CudaDensePageTransition,
        sources: tuple[StateRef, ...],
        targets: tuple[StateRef, ...],
        target_tensors: tuple[tuple[torch.Tensor, ...], ...],
        stream: torch.cuda.Stream,
    ) -> None:
        """Prevalidate one unsubmitted full-root COW publication."""

        with self._lock:
            self._require_stream(stream)
            state = self._prepared_transition(transition)
            if stream.cuda_stream != state.stream.cuda_stream:
                raise RuntimeError(
                    "CUDA transition submission uses another prepared stream"
                )
            if state.batch.sources != sources or state.batch.targets != targets:
                raise ValueError("CUDA transition differs from logical root versions")
            self._pools.require_full_root_transition(
                state.batch.target_page_ids,
                target_tensors,
            )

    def submit_full_root_transition(
        self,
        transition: CudaDensePageTransition,
        target_tensors: tuple[tuple[torch.Tensor, ...], ...],
        stream: torch.cuda.Stream,
    ) -> CudaDensePagePublication:
        """Copy full private roots, fence them, then atomically publish."""

        with self._lock:
            self._require_stream(stream)
            state = self._prepared_transition(transition)
            if stream.cuda_stream != state.stream.cuda_stream:
                raise RuntimeError(
                    "CUDA transition submission uses another prepared stream"
                )
            self._pools.require_full_root_transition(
                state.batch.target_page_ids,
                target_tensors,
            )
            self._ledger.mark_submitted(state.batch)
            try:
                self._pools.submit_full_root_transition(
                    target_page_ids=state.batch.target_page_ids,
                    target_tensors=target_tensors,
                    stream=stream,
                )
                readiness = self._record_readiness(stream)
                roots = self._ledger.publish_batch(state.batch, readiness)
                self._prepared_transitions.pop(transition.transaction_id)
                return CudaDensePagePublication(tuple(root.ref for root in roots))
            except Exception:
                self._poison()
                raise

    def abort_unsubmitted_transition(
        self,
        transition: CudaDensePageTransition,
    ) -> None:
        with self._lock:
            self._abort_unsubmitted_transition(transition)

    def _abort_unsubmitted_transition(
        self,
        transition: CudaDensePageTransition,
    ) -> None:
        state = self._prepared_transition(transition)
        self._ledger.abort_unsubmitted_batch(state.batch)
        self._prepared_transitions.pop(transition.transaction_id)

    def bind_gradients(
        self,
        values: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream,
    ) -> CudaDensePageGradientBatch:
        with self._lock:
            self._require_stream(stream)
            if (
                type(values) is not tuple
                or not values
                or any(type(value) is not torch.Tensor for value in values)
            ):
                raise ValueError("CUDA gradient batch requires tensors")
            token = CudaDensePageGradientBatch(self._next_gradient_batch_id)
            self._next_gradient_batch_id += 1
            self._gradient_batches[token.batch_id] = _BoundGradientState(
                token,
                values,
                stream,
            )
            return token

    def discard_gradients(self, token: CudaDensePageGradientBatch) -> None:
        with self._lock:
            self._bound_gradients(token)
            self._gradient_batches.pop(token.batch_id)

    def pin_exact(
        self,
        refs: tuple[StateRef, ...],
        stream: torch.cuda.Stream,
    ) -> CudaDensePageVersionPin:
        with self._lock:
            self._require_stream(stream)
            return self._ledger.pin_exact(refs, stream)

    def release_pin(
        self,
        pin: CudaDensePageVersionPin,
        stream: torch.cuda.Stream,
    ) -> CudaDensePageRelease:
        with self._lock:
            self._require_stream(stream)
            event = torch.cuda.Event(blocking=False)
            event.record(stream)
            completion = CudaDensePageFence(
                self._next_completion_fence_id,
                self._selected.device_ordinal,
                stream,
                event,
            )
            self._ledger.release_pin(pin, completion)
            self._next_completion_fence_id += 1
            receipt = CudaDensePageRelease(
                self._next_release_id,
                pin,
            )
            self._next_release_id += 1
            self._release_receipts[receipt.release_id] = receipt
            return receipt

    def wait_release(self, token: CudaDensePageRelease) -> None:
        with self._lock:
            issued = self._issued_release(token)
            self._ledger.wait_pin_release(issued.pin)

    def release_complete(self, token: CudaDensePageRelease) -> bool:
        with self._lock:
            issued = self._issued_release(token)
            return self._ledger.pin_release_complete(issued.pin)

    def snapshot(
        self,
        pin: CudaDensePageVersionPin,
        ref: StateRef,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, ...]:
        with self._lock:
            return self._snapshot(pin, ref, stream)

    def pinned_field_rows(
        self,
        pin: CudaDensePageVersionPin,
        field_id: LayoutFieldId,
        stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        with self._lock:
            self._require_stream(stream)
            page_ids = self._ledger.pinned_page_ids(pin, stream)
            return self._pools.field_rows(page_ids, field_id, stream)

    def _snapshot(
        self,
        pin: CudaDensePageVersionPin,
        ref: StateRef,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, ...]:
        self._require_stream(stream)
        try:
            pin_position = pin.refs.index(ref)
        except ValueError as exc:
            raise RuntimeError("CUDA snapshot root is not held by its pin") from exc
        page_ids = self._ledger.pinned_page_ids(pin, stream)
        page_id = page_ids[pin_position]
        return self._pools.snapshot(page_id, stream)

    def retire_batch(self, refs: tuple[StateRef, ...]) -> None:
        with self._lock:
            self._require_live()
            self._ledger.retire_batch(refs)

    def collect_completed(self) -> None:
        with self._lock:
            self._require_live()
            self._ledger.collect_completed()

    def _prepared_admission(
        self,
        token: CudaDensePagePreparedAdmission,
    ) -> _PreparedAdmissionState:
        try:
            state = self._prepared_admissions[token.transaction_id]
        except KeyError as exc:
            raise RuntimeError(
                "CUDA admission token is unknown or consumed"
            ) from exc
        if state.token is not token:
            raise RuntimeError("CUDA admission token differs from the issued token")
        return state

    def _prepared_transition(
        self,
        token: CudaDensePageTransition,
    ) -> _PreparedTransitionState:
        try:
            state = self._prepared_transitions[token.transaction_id]
        except KeyError as exc:
            raise RuntimeError(
                "CUDA transition token is unknown or consumed"
            ) from exc
        if state.token is not token:
            raise RuntimeError("CUDA transition token differs from the issued token")
        return state

    def _bound_gradients(
        self,
        token: CudaDensePageGradientBatch,
    ) -> _BoundGradientState:
        try:
            state = self._gradient_batches[token.batch_id]
        except KeyError as exc:
            raise RuntimeError(
                "CUDA gradient token is unknown or consumed"
            ) from exc
        if state.token is not token:
            raise RuntimeError("CUDA gradient token differs from the issued token")
        return state

    def _issued_release(
        self,
        token: CudaDensePageRelease,
    ) -> CudaDensePageRelease:
        self._require_live()
        try:
            issued = self._release_receipts[token.release_id]
        except KeyError as exc:
            raise RuntimeError(
                "CUDA release receipt is unknown or discarded"
            ) from exc
        if issued is not token:
            raise RuntimeError(
                "CUDA release receipt differs from the issued receipt"
            )
        return issued

    def _require_stream(self, stream: torch.cuda.Stream) -> None:
        self._require_live()
        if type(stream) is not torch.cuda.Stream or stream.device != self._pools.device:
            raise RuntimeError("CUDA dense-page stream uses another device")

    def _poison(self) -> None:
        self._ledger.poison()
        self._poisoned = True

    def _require_live(self) -> None:
        if self._poisoned or self._ledger.poisoned:
            self._poisoned = True
            raise RuntimeError("CUDA dense-page provider is poisoned")

    def _record_readiness(
        self,
        stream: torch.cuda.Stream,
    ) -> CudaDensePageFence:
        event = torch.cuda.Event(blocking=False)
        event.record(stream)
        return CudaDensePageFence(
            0,
            self._selected.device_ordinal,
            stream,
            event,
        )

__all__ = [
    "CudaDensePageAdmission",
    "CudaDensePageGradientBatch",
    "CudaDensePageInitialRoot",
    "CudaDensePagePublication",
    "CudaDensePagePreparedAdmission",
    "CudaDensePageStateProvider",
    "CudaDensePageTransition",
]
