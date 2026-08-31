"""Request-scoped trainable state composed from one arena and one adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar

import torch

from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
    RequestKeyField,
)
from tllm.contracts.request_state import (
    IndexedRequestIdentityBatch,
    IndexedRequestIdentityRow,
    RequestIdentityBatch,
    RequestStateAcquisition,
    RequestStateAsyncPin,
    RequestStateEffectAudit,
    RequestStateKey,
    RequestStateLease,
    RequestStateLossBatch,
    RequestStateSelectionBatch,
    RequestStateSelectionRow,
    RequestStateVersion,
    StateEffectOperation,
    StateEffectReceipt,
    StateSlotEffect,
    StateSlotRef,
)
from tllm.contracts.state_ref import LayoutId, VersionId
from tllm.runtime.request_slot_arena import (
    RequestSlotAdmissionPlan,
    RequestSlotArena,
    RequestSlotPinPlan,
)
from tllm.runtime.request_slot_adapter import (
    CompletionFence,
    RequestSlotAdapter,
)
from tllm.runtime.fixed_batched_slot_adapter import (
    FixedBatchedAdmissionPreparation,
    FixedBatchedSlotAdapter,
    FixedBatchedSlotView,
)
from tllm.runtime.paged_state import StateResourceId
from tllm.runtime.resident_module_adapter import (
    RequestStateModuleGroup,
    ResidentAdmissionPreparation,
    ResidentModuleAdapter,
    ResidentSlotView,
)
from tllm.runtime.stable_packed_state import (
    FrozenPackedTensorLayout,
    StablePackedStateView,
    StablePackedStateViewArena,
    StateViewCompletionFence,
)


ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


@dataclass(frozen=True, slots=True)
class _ResidentPhysical(Generic[ModuleT]):
    adapter: ResidentModuleAdapter[ModuleT]


@dataclass(frozen=True, slots=True)
class _FixedBatchedPhysical(Generic[ModuleT]):
    adapter: FixedBatchedSlotAdapter[ModuleT]


_RequestStatePhysical = (
    _ResidentPhysical[ModuleT] | _FixedBatchedPhysical[ModuleT]
)


@dataclass(frozen=True, slots=True)
class _ResidentPreparation:
    value: ResidentAdmissionPreparation


@dataclass(frozen=True, slots=True)
class _FixedBatchedPreparation:
    value: FixedBatchedAdmissionPreparation


_PhysicalPreparation = _ResidentPreparation | _FixedBatchedPreparation
RequestStateSelected = ResidentSlotView[ModuleT] | FixedBatchedSlotView[ModuleT]


class RequestTrainableStateStore(Generic[ModuleT]):
    """Compose logical request slots with one selected physical adapter.

    The arena is the only logical authority. This class holds a transaction
    lock across arena reservation, physical preparation/commit, and logical
    commit; it does not mirror any key, generation, version, pin, or receipt.
    """

    __slots__ = (
        "_arena",
        "_physical",
        "_stable_view_arenas",
        "_lock",
    )

    def __init__(
        self,
        *,
        resource: str,
        key_fields: tuple[RequestKeyField, ...],
        capacity: int,
        epoch: int,
        device: torch.device,
        module_factory: Callable[[], ModuleT],
        optimizer_factory: Callable[[ModuleT], torch.optim.Optimizer],
    ) -> None:
        self._arena = RequestSlotArena(
            resource=resource,
            key_fields=key_fields,
            capacity=capacity,
            epoch=epoch,
            device=device,
        )
        self._physical: _RequestStatePhysical[ModuleT] = _ResidentPhysical(
            ResidentModuleAdapter(
                capacity=capacity,
                device=device,
                module_factory=module_factory,
                optimizer_factory=optimizer_factory,
            )
        )
        self._stable_view_arenas: list[StablePackedStateViewArena] = []
        self._lock = RLock()

    @classmethod
    def fixed_batched(
        cls,
        *,
        resource: str,
        layout_id: LayoutId,
        key_fields: tuple[RequestKeyField, ...],
        epoch: int,
        bank: ModuleT,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> "RequestTrainableStateStore[ModuleT]":
        store = cls.__new__(cls)
        adapter = FixedBatchedSlotAdapter(
            bank,
            resource=StateResourceId(resource),
            layout_id=layout_id,
            epoch=epoch,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        store._arena = RequestSlotArena(
            resource=resource,
            key_fields=key_fields,
            capacity=adapter.capacity,
            epoch=epoch,
            device=adapter.device,
        )
        store._physical = _FixedBatchedPhysical(adapter)
        store._stable_view_arenas = []
        store._lock = RLock()
        return store

    @property
    def resource(self) -> str:
        return self._arena.resource

    @property
    def capacity(self) -> int:
        return self._arena.capacity

    @property
    def epoch(self) -> int:
        return self._arena.epoch

    @property
    def device(self) -> torch.device:
        return self._adapter.device

    @property
    def _adapter(self) -> RequestSlotAdapter:
        match self._physical:
            case _ResidentPhysical(adapter=adapter):
                return adapter
            case _FixedBatchedPhysical(adapter=adapter):
                return adapter

    def acquire(
        self, identities: tuple[RequestIdentity, ...]
    ) -> RequestStateAcquisition:
        return self._acquire(
            identities,
            tuple(range(len(identities))),
        )

    def acquire_indexed(
        self, batch: IndexedRequestIdentityBatch
    ) -> RequestStateAcquisition:
        return self._acquire(
            batch.identities,
            tuple(row.canonical_row_index for row in batch.rows),
        )

    def _acquire(
        self,
        identities: tuple[RequestIdentity, ...],
        canonical_row_indices: tuple[int, ...],
    ) -> RequestStateAcquisition:
        with self._lock:
            self._collect_stable_views()
            plan = self._arena.plan_admission(
                identities,
                canonical_row_indices,
            )
            try:
                preparation = self._prepare_admission(plan)
            except Exception:
                self._arena.abort_admission(plan, poison=False)
                raise
            try:
                self._commit_admission(plan, preparation)
            except Exception:
                self._adapter.poison()
                self._arena.abort_admission(plan, poison=True)
                raise
            try:
                return self._arena.commit_admission(plan)
            except Exception:
                self._fail_stop()
                raise

    def _prepare_admission(
        self,
        plan: RequestSlotAdmissionPlan,
    ) -> _PhysicalPreparation:
        match self._physical:
            case _ResidentPhysical(adapter=adapter):
                return _ResidentPreparation(adapter.prepare_admission(plan))
            case _FixedBatchedPhysical(adapter=adapter):
                return _FixedBatchedPreparation(
                    adapter.prepare_admission(plan)
                )

    def _commit_admission(
        self,
        plan: RequestSlotAdmissionPlan,
        preparation: _PhysicalPreparation,
    ) -> None:
        match self._physical, preparation:
            case (
                _ResidentPhysical(adapter=adapter),
                _ResidentPreparation(value=value),
            ):
                adapter.commit_admission(plan, value)
            case (
                _FixedBatchedPhysical(adapter=adapter),
                _FixedBatchedPreparation(value=value),
            ):
                adapter.commit_admission(plan, value)
            case _:
                raise RuntimeError(
                    "request-state physical admission variant changed"
                )

    def selected(
        self, selection: RequestStateSelectionBatch
    ) -> RequestStateSelected[ModuleT]:
        with self._lock:
            self._arena.validate_selection(selection, audited=True)
            match self._physical:
                case _ResidentPhysical(adapter=adapter):
                    return adapter.selected(selection)
                case _FixedBatchedPhysical(adapter=adapter):
                    return adapter.selected(
                        selection,
                        self._arena.selection_versions(selection),
                    )

    def make_stable_view_arena(
        self,
        *,
        view_epoch: int,
        layout: FrozenPackedTensorLayout,
        buffer_count: int,
    ) -> StablePackedStateViewArena:
        with self._lock:
            match self._physical:
                case _FixedBatchedPhysical(adapter=adapter):
                    views = StablePackedStateViewArena(
                        adapter.version_roots,
                        resource=StateResourceId(self.resource),
                        state_epoch=self.epoch,
                        view_epoch=view_epoch,
                        layout=layout,
                        buffer_count=buffer_count,
                    )
                    self._stable_view_arenas.append(views)
                    return views
                case _ResidentPhysical():
                    raise RuntimeError(
                        "resident request state cannot produce stable packed views"
                    )

    def create_stable_view_synchronous(
        self,
        views: StablePackedStateViewArena,
        selection: RequestStateSelectionBatch,
        tensors: tuple[torch.Tensor, ...],
    ) -> StablePackedStateView:
        with self._lock:
            self._validate_stable_view_arena(views)
            versions = self._arena.selection_versions(selection)
            return views.create_synchronous(selection, versions, tensors)

    def create_stable_view_cuda(
        self,
        views: StablePackedStateViewArena,
        selection: RequestStateSelectionBatch,
        tensors: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream,
    ) -> StablePackedStateView:
        with self._lock:
            self._validate_stable_view_arena(views)
            versions = self._arena.selection_versions(selection)
            return views.create_cuda(selection, versions, tensors, stream)

    def _validate_stable_view_arena(
        self,
        views: StablePackedStateViewArena,
    ) -> None:
        match self._physical:
            case _FixedBatchedPhysical(adapter=adapter):
                if (
                    views.version_roots is not adapter.version_roots
                    or views.resource != StateResourceId(self.resource)
                    or views.state_epoch != self.epoch
                ):
                    raise RuntimeError(
                        "stable packed view belongs to another physical state"
                    )
            case _ResidentPhysical():
                raise RuntimeError(
                    "resident request state cannot produce stable packed views"
                )

    def selection_versions(
        self, selection: RequestStateSelectionBatch
    ) -> tuple[RequestStateVersion, ...]:
        with self._lock:
            return self._arena.selection_versions(selection)

    def read_receipt(
        self, selection: RequestStateSelectionBatch
    ) -> StateEffectReceipt:
        with self._lock:
            match self._physical:
                case _FixedBatchedPhysical(adapter=adapter) if (
                    self.device.type == "cpu"
                ):
                    adapter.complete_reads_synchronous(selection)
                case _ResidentPhysical() | _FixedBatchedPhysical():
                    pass
            return self._arena.read_receipt(selection)

    def optimize(
        self,
        selection: RequestStateSelectionBatch,
        losses: tuple[torch.Tensor, ...],
    ) -> StateEffectReceipt:
        if len(losses) != len(selection.leases):
            raise ValueError("request state losses must align with unique leases")
        if any(loss.numel() != 1 for loss in losses):
            raise ValueError("request state optimizer loss must be scalar")
        with self._lock:
            self._collect_stable_views()
            plan = self._arena.plan_transition(selection)
            try:
                match self._physical:
                    case _ResidentPhysical(adapter=adapter):
                        adapter.execute_transition(plan, losses)
                    case _FixedBatchedPhysical(adapter=adapter):
                        adapter.execute_transition(plan, losses)
            except Exception:
                self._adapter.poison()
                self._arena.abort_transition(plan, poison=True)
                raise
            try:
                return self._arena.commit_transition(plan)
            except Exception:
                self._fail_stop()
                raise

    def release_selection(self, selection: RequestStateSelectionBatch) -> None:
        with self._lock:
            self._arena.release_selection(selection)

    def pin_synchronous(
        self, selection: RequestStateSelectionBatch
    ) -> RequestStateAsyncPin:
        with self._lock:
            plan = self._arena.plan_pin(selection)
            try:
                fence = self._adapter.record_synchronous_fence()
            except Exception:
                self._arena.abort_pin(plan, poison=False)
                raise
            return self._commit_pin(plan, fence)

    def pin_cuda(
        self,
        selection: RequestStateSelectionBatch,
        stream: torch.cuda.Stream,
    ) -> RequestStateAsyncPin:
        with self._lock:
            plan = self._arena.plan_pin(selection)
            try:
                fence = self._adapter.record_cuda_fence(stream)
            except Exception:
                self._arena.abort_pin(plan, poison=False)
                raise
            return self._commit_pin(plan, fence)

    def _commit_pin(
        self,
        plan: RequestSlotPinPlan,
        fence: CompletionFence,
    ) -> RequestStateAsyncPin:
        try:
            self._adapter.commit_fence(plan, fence)
        except Exception:
            self._adapter.poison()
            self._arena.abort_pin(plan, poison=True)
            raise
        try:
            return self._arena.commit_pin(plan)
        except Exception:
            self._fail_stop()
            raise

    def release_async_pin(self, pin: RequestStateAsyncPin) -> None:
        with self._lock:
            self._arena.validate_pin(pin)
            try:
                complete = self._adapter.fence_complete(pin)
            except Exception:
                self._fail_stop()
                raise
            if not complete:
                raise RuntimeError("request state CUDA completion is still pending")
            try:
                self._release_completed_pin(pin)
            except Exception:
                self._fail_stop()
                raise

    def async_pin_complete(self, pin: RequestStateAsyncPin) -> bool:
        with self._lock:
            self._arena.validate_pin(pin)
            try:
                return self._adapter.fence_complete(pin)
            except Exception:
                self._fail_stop()
                raise

    def order_async_pin(
        self,
        pin: RequestStateAsyncPin,
        stream: torch.cuda.Stream,
    ) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("CUDA request state pin has no completion event")
        if stream.device != self.device:
            raise RuntimeError("request state ordering stream uses another device")
        with self._lock:
            self._arena.validate_pin(pin)
            try:
                self._adapter.order_fence(pin, stream)
                self._release_completed_pin(pin)
            except Exception:
                self._fail_stop()
                raise

    def wait_async_pin(self, pin: RequestStateAsyncPin) -> None:
        with self._lock:
            self._arena.validate_pin(pin)
            try:
                self._adapter.wait_fence(pin)
                self._release_completed_pin(pin)
            except Exception:
                self._fail_stop()
                raise

    def _release_completed_pin(self, pin: RequestStateAsyncPin) -> None:
        self._adapter.discard_fence(pin)
        self._arena.release_pin(pin)

    def retire(self, batch: RequestFinishedBatch) -> StateEffectReceipt:
        with self._lock:
            self._collect_stable_views()
            receipt = self._arena.retire(batch)
            try:
                self._adapter.retire(receipt)
            except Exception:
                self._fail_stop()
                raise
            return receipt

    def versions(self) -> tuple[RequestStateVersion, ...]:
        with self._lock:
            return self._arena.versions()

    def consume_receipt(self, receipt: StateEffectReceipt) -> None:
        with self._lock:
            self._arena.consume_receipt(receipt)

    def replace_epoch(self, epoch: int) -> None:
        with self._lock:
            self._arena.validate_epoch_replacement(epoch)
            self._collect_stable_views()
            if any(
                views.free_buffers != views.buffer_count
                for views in self._stable_view_arenas
            ):
                raise RuntimeError(
                    "cannot replace request-state epoch with live stable views"
                )
            try:
                self._adapter.replace_epoch(epoch)
                self._arena.replace_epoch(epoch)
                self._stable_view_arenas.clear()
            except Exception:
                self._fail_stop()
                raise

    def _collect_stable_views(self) -> None:
        for views in self._stable_view_arenas:
            views.collect_completed()

    def fail_stop(self) -> None:
        with self._lock:
            self._fail_stop()

    def _fail_stop(self) -> None:
        self._adapter.poison()
        self._arena.poison()


@dataclass(frozen=True, slots=True)
class _StagedIdentities:
    identities: RequestIdentityBatch | IndexedRequestIdentityBatch


@dataclass(frozen=True, slots=True)
class _StagedSelection:
    identities: RequestIdentityBatch | IndexedRequestIdentityBatch
    selection: RequestStateSelectionBatch


_StagedRequestStateStep = _StagedIdentities | _StagedSelection


class _RequestStateRuntimeCore(Generic[ModuleT]):
    """Shared mutable authority behind locally and composition-qualified views."""

    __slots__ = (
        "store",
        "steps",
        "transition_receipts",
        "completion_pins",
        "stable_view_counts",
        "lock",
    )

    def __init__(self, store: RequestTrainableStateStore[ModuleT]) -> None:
        self.store = store
        self.steps: dict[int, _StagedRequestStateStep] = {}
        self.transition_receipts: dict[int, StateEffectReceipt] = {}
        self.completion_pins: dict[int, list[RequestStateAsyncPin]] = {}
        self.stable_view_counts: dict[int, int] = {}
        self.lock = RLock()


class RequestStateRuntimeService(Generic[ModuleT]):
    """Runtime authority for staged request-state execution and auditing."""

    __slots__ = ("_core", "_resource")

    def __init__(self, store: RequestTrainableStateStore[ModuleT]) -> None:
        self._core = _RequestStateRuntimeCore(store)
        self._resource = store.resource

    @classmethod
    def _for_core(
        cls,
        core: _RequestStateRuntimeCore[ModuleT],
        resource: str,
    ) -> "RequestStateRuntimeService[ModuleT]":
        service = cls.__new__(cls)
        service._core = core
        service._resource = resource
        return service

    @property
    def _store(self) -> RequestTrainableStateStore[ModuleT]:
        return self._core.store

    @property
    def _steps(self) -> dict[int, _StagedRequestStateStep]:
        return self._core.steps

    @property
    def _transition_receipts(self) -> dict[int, StateEffectReceipt]:
        return self._core.transition_receipts

    @property
    def _completion_pins(self) -> dict[int, list[RequestStateAsyncPin]]:
        return self._core.completion_pins

    @property
    def _stable_view_counts(self) -> dict[int, int]:
        return self._core.stable_view_counts

    @property
    def _lock(self) -> RLock:
        return self._core.lock

    @property
    def resource(self) -> str:
        return self._resource

    @property
    def epoch(self) -> int:
        return self._store.epoch

    @property
    def capacity(self) -> int:
        return self._store.capacity

    @property
    def device(self) -> torch.device:
        return self._store.device

    def versions(self) -> tuple[RequestStateVersion, ...]:
        with self._lock:
            return self._store.versions()

    def for_signed_resource(
        self, resource: str
    ) -> "RequestStateRuntimeService[ModuleT]":
        if not resource:
            raise ValueError("signed request-state resource must be non-empty")
        with self._lock:
            if (
                self._steps
                or self._transition_receipts
                or self._completion_pins
                or self._stable_view_counts
            ):
                raise RuntimeError(
                    "cannot qualify request-state service while it is active"
                )
            if resource == self.resource:
                return self
            return self._for_core(self._core, resource)

    def stable_view_service(
        self,
        *,
        view_epoch: int,
        layout: FrozenPackedTensorLayout,
        buffer_count: int,
    ) -> "RequestStateStableViewService[ModuleT]":
        with self._lock:
            if (
                self._steps
                or self._transition_receipts
                or self._completion_pins
                or self._stable_view_counts
            ):
                raise RuntimeError(
                    "stable view service must be created before request execution"
                )
            return RequestStateStableViewService(
                self._core,
                self._resource,
                self._store.make_stable_view_arena(
                    view_epoch=view_epoch,
                    layout=layout,
                    buffer_count=buffer_count,
                ),
            )

    def stage_identities(self, step_id: int, batch: RequestIdentityBatch) -> None:
        self._stage_identity_batch(step_id, batch)

    def stage_indexed_identities(
        self, step_id: int, batch: IndexedRequestIdentityBatch
    ) -> None:
        self._stage_identity_batch(step_id, batch)

    def _stage_identity_batch(
        self,
        step_id: int,
        batch: RequestIdentityBatch | IndexedRequestIdentityBatch,
    ) -> None:
        if step_id < 0:
            raise ValueError("request-state step id must be non-negative")
        with self._lock:
            self._collect_completed_pins()
            if step_id not in self._steps:
                self._steps[step_id] = _StagedIdentities(batch)
                return
            if self._steps[step_id].identities != batch:
                raise RuntimeError(
                    "request-state step identity batch differs from its staged batch"
                )

    def _acquire_selection(self, step_id: int) -> RequestStateSelectionBatch:
        with self._lock:
            try:
                staged = self._steps[step_id]
            except KeyError as exc:
                raise RuntimeError(
                    "request-state identities must be staged before selection acquisition"
                ) from exc
            match staged:
                case _StagedSelection(selection=selection):
                    return selection
                case _StagedIdentities(
                    identities=RequestIdentityBatch(identities=identities)
                ):
                    acquired = self._store.acquire(identities)
                case _StagedIdentities(
                    identities=IndexedRequestIdentityBatch() as indexed
                ):
                    acquired = self._store.acquire_indexed(indexed)
                case _StagedIdentities():
                    raise AssertionError(
                        "request-state staged identity variant is not closed"
                    )
            self._validate_admission(acquired)
            self._store.consume_receipt(acquired.admission)
            self._steps[step_id] = _StagedSelection(
                staged.identities,
                acquired.selection,
            )
            return acquired.selection

    def selection_for_step(
        self,
        step_id: int,
        stream: torch.cuda.Stream | None = None,
    ) -> RequestStateSelectionBatch:
        selection = self.selection_value_for_step(step_id, stream)
        if selection is None:
            raise RuntimeError("request-state identity batch is empty")
        return selection

    def selection_value_for_step(
        self,
        step_id: int,
        stream: torch.cuda.Stream | None = None,
    ) -> RequestStateSelectionBatch | None:
        with self._lock:
            try:
                staged = self._steps[step_id]
            except KeyError as exc:
                raise RuntimeError(
                    "request-state identities must be staged before selection acquisition"
                ) from exc
            if self._store.device.type == "cuda":
                if stream is None:
                    raise RuntimeError(
                        "CUDA request-state selection requires its signed execution stream"
                    )
                self._order_completion_pins(stream)
            if not staged.identities.identities:
                return None
        return self._acquire_selection(step_id)

    def before(
        self,
        operation: StateEffectOperation,
        selection: RequestStateSelectionBatch,
        step_id: int,
    ) -> RequestStateEffectAudit:
        with self._lock:
            self._require_staged_selection(step_id, selection)
            self._validate_operation(operation, selection)
            selected = tuple(
                StateSlotEffect(
                    version.key,
                    version.slot,
                    version.lineage_id,
                    version.version,
                    version.version,
                )
                for version in self._store.selection_versions(selection)
            )
            return RequestStateEffectAudit(
                resource=self.resource,
                epoch=self.epoch,
                step_id=step_id,
                selection=selection,
                operation=operation,
                selected_before=selected,
                store_before=self._store.versions(),
            )

    def optimize(
        self,
        selection: RequestStateSelectionBatch,
        losses: tuple[torch.Tensor, ...],
    ) -> StateEffectReceipt:
        with self._lock:
            self._require_selection(selection)
            if selection.selection_id in self._transition_receipts:
                raise RuntimeError(
                    "request-state selection already has an unaudited transition"
                )
            receipt = self._store.optimize(selection, losses)
            self._transition_receipts[selection.selection_id] = receipt
            return receipt

    def selected(
        self, selection: RequestStateSelectionBatch
    ) -> RequestStateSelected[ModuleT]:
        with self._lock:
            self._require_selection(selection)
            return self._store.selected(selection)

    def after(self, audit: RequestStateEffectAudit) -> None:
        with self._lock:
            self._require_staged_selection(audit.step_id, audit.selection)
            if audit.resource != self.resource or audit.epoch != self.epoch:
                raise RuntimeError(
                    "request-state audit belongs to another resource epoch"
                )
            if audit.operation is StateEffectOperation.READ:
                receipt = self._store.read_receipt(audit.selection)
            elif audit.operation is StateEffectOperation.TRANSITION:
                try:
                    receipt = self._transition_receipts[
                        audit.selection.selection_id
                    ]
                except KeyError as exc:
                    raise RuntimeError(
                        "request-state transition has no service-owned effect receipt"
                    ) from exc
            else:
                raise RuntimeError(
                    "request-state runtime supports only strict read and "
                    "transition effects"
                )
            self._validate_effect_receipt(audit, receipt)
            self._store.consume_receipt(receipt)
            if audit.operation is StateEffectOperation.TRANSITION:
                self._transition_receipts.pop(audit.selection.selection_id)

    def record_completion(
        self,
        step_id: int,
        selection: RequestStateSelectionBatch,
        stream: torch.cuda.Stream,
    ) -> None:
        with self._lock:
            self._require_staged_selection(step_id, selection)
            if self._store.device.type != "cuda":
                raise RuntimeError(
                    "CPU request state cannot record a CUDA completion"
                )
            pin = self._store.pin_cuda(selection, stream)
            self._completion_pins.setdefault(step_id, []).append(pin)

    def finish_step(self, step_id: int) -> None:
        with self._lock:
            if step_id not in self._steps:
                return
            staged = self._steps[step_id]
            match staged:
                case _StagedIdentities():
                    self._steps.pop(step_id)
                    return
                case _StagedSelection(selection=selection):
                    pass
            if selection.selection_id in self._transition_receipts:
                raise RuntimeError(
                    "request-state step retires with an unaudited transition receipt"
                )
            if self._stable_view_counts.get(step_id, 0):
                raise RuntimeError(
                    "request-state step has an uncompleted stable packed view"
                )
            self._store.release_selection(selection)
            self._steps.pop(step_id)

    def retire_requests(
        self, batch: RequestFinishedBatch
    ) -> StateEffectReceipt:
        with self._lock:
            if self._steps or self._transition_receipts:
                raise RuntimeError(
                    "request-state retirement requires no active step, selection, "
                    "or unaudited receipt"
                )
            if self._stable_view_counts:
                raise RuntimeError(
                    "request-state retirement has an uncompleted stable view"
                )
            for step_id in tuple(sorted(self._completion_pins)):
                pins = self._completion_pins[step_id]
                while pins:
                    self._store.wait_async_pin(pins[0])
                    pins.pop(0)
                self._completion_pins.pop(step_id)
            receipt = self._store.retire(batch)
            if (
                receipt.resource != self._store.resource
                or receipt.epoch != self.epoch
                or receipt.operation is not StateEffectOperation.RETIRE
                or receipt.selection_id != 0
                or any(
                    effect.before_version != effect.after_version
                    for effect in receipt.slots
                )
                or len({effect.slot for effect in receipt.slots})
                != len(receipt.slots)
            ):
                raise RuntimeError(
                    "request-state retirement receipt has the wrong identity"
                )
            self._store.consume_receipt(receipt)
            return receipt

    def _collect_completed_pins(self) -> None:
        for step_id in tuple(sorted(self._completion_pins)):
            pending: list[RequestStateAsyncPin] = []
            for pin in self._completion_pins[step_id]:
                if self._store.async_pin_complete(pin):
                    self._store.release_async_pin(pin)
                else:
                    pending.append(pin)
            if pending:
                self._completion_pins[step_id] = pending
            else:
                self._completion_pins.pop(step_id)

    def _order_completion_pins(self, stream: torch.cuda.Stream) -> None:
        for step_id in tuple(sorted(self._completion_pins)):
            pins = self._completion_pins[step_id]
            while pins:
                self._store.order_async_pin(pins[0], stream)
                pins.pop(0)
            self._completion_pins.pop(step_id)

    def _require_selection(self, selection: RequestStateSelectionBatch) -> None:
        matching_steps: list[int] = []
        for step_id, staged in self._steps.items():
            match staged:
                case _StagedSelection(selection=owned) if owned is selection:
                    matching_steps.append(step_id)
                case _:
                    pass
        if len(matching_steps) != 1:
            raise RuntimeError(
                "request-state selection is not owned by this runtime service"
            )

    def _require_staged_selection(
        self,
        step_id: int,
        selection: RequestStateSelectionBatch,
    ) -> None:
        try:
            staged = self._steps[step_id]
        except KeyError as exc:
            raise RuntimeError("request-state step is not staged") from exc
        match staged:
            case _StagedSelection(selection=owned) if owned is selection:
                return
            case _:
                raise RuntimeError(
                    "request-state audit selection differs from the staged "
                    "step selection"
                )

    def _validate_admission(self, acquisition: RequestStateAcquisition) -> None:
        selection = acquisition.selection
        receipt = acquisition.admission
        if (
            receipt.resource != self._store.resource
            or receipt.epoch != self.epoch
            or receipt.operation is not StateEffectOperation.ADMIT
            or receipt.selection_id != selection.selection_id
        ):
            raise RuntimeError(
                "request-state admission receipt has the wrong identity"
            )
        leases = tuple(row.lease for row in selection.rows)
        lease_by_slot = {lease.slot: lease for lease in selection.leases}
        if any(
            lease.slot not in lease_by_slot
            or lease_by_slot[lease.slot] != lease
            for lease in leases
        ):
            raise RuntimeError(
                "request-state admission selection has an invalid slot lease"
            )
        for effect in receipt.slots:
            if (
                effect.slot not in lease_by_slot
                or effect.key != lease_by_slot[effect.slot].key
                or effect.lineage_id
                != lease_by_slot[effect.slot].lineage_id
                or effect.before_version != VersionId(0)
                or effect.after_version != VersionId(0)
            ):
                raise RuntimeError(
                    "request-state admission receipt differs from selection"
                )

    def _validate_operation(
        self,
        operation: StateEffectOperation,
        selection: RequestStateSelectionBatch,
    ) -> None:
        if operation not in (
            StateEffectOperation.READ,
            StateEffectOperation.TRANSITION,
        ):
            raise RuntimeError(
                "request-state runtime supports only strict read and "
                "transition effects"
            )
        if (
            selection.resource != self._store.resource
            or selection.epoch != self.epoch
        ):
            raise RuntimeError(
                "request-state selection belongs to another resource epoch"
            )

    def _validate_effect_receipt(
        self,
        audit: RequestStateEffectAudit,
        receipt: StateEffectReceipt,
    ) -> None:
        operation = (
            StateEffectOperation.READ
            if audit.operation is StateEffectOperation.READ
            else StateEffectOperation.TRANSITION
        )
        selection = audit.selection
        if (
            receipt.resource != self._store.resource
            or receipt.epoch != audit.epoch
            or receipt.operation is not operation
            or receipt.selection_id != selection.selection_id
        ):
            raise RuntimeError(
                "request-state effect receipt has the wrong identity"
            )
        if tuple(effect.slot for effect in receipt.slots) != tuple(
            effect.slot for effect in audit.selected_before
        ):
            raise RuntimeError(
                "request-state effect receipt slot generation differs"
            )
        current = {version.slot: version for version in self._store.versions()}
        before_all = {
            version.slot: version for version in audit.store_before
        }
        if set(current) != set(before_all):
            raise RuntimeError(
                "request-state slot identity changed while a Region effect executed"
            )
        selected_slots = {effect.slot for effect in audit.selected_before}
        if len(selected_slots) != len(audit.selected_before):
            raise RuntimeError(
                "request-state selection contains duplicate slot leases"
            )
        expected_delta = 0 if operation is StateEffectOperation.READ else 1
        for before, effect in zip(
            audit.selected_before,
            receipt.slots,
            strict=True,
        ):
            expected_after = VersionId(
                int(before.before_version) + expected_delta
            )
            if before.slot not in current:
                raise RuntimeError(
                    "request-state selected slot violated its signed "
                    "before/after effect"
                )
            current_version = current[before.slot]
            if (
                effect.key != before.key
                or effect.slot != before.slot
                or effect.lineage_id != before.lineage_id
                or effect.before_version != before.before_version
                or effect.after_version != expected_after
                or current_version.key != before.key
                or current_version.lineage_id != before.lineage_id
                or current_version.version != expected_after
            ):
                raise RuntimeError(
                    "request-state selected slot violated its signed "
                    "before/after effect"
                )
        for slot, before in before_all.items():
            if slot in selected_slots:
                continue
            current_version = current[slot]
            if (
                current_version.key != before.key
                or current_version.lineage_id != before.lineage_id
                or current_version.version != before.version
            ):
                raise RuntimeError(
                    "request-state unselected slot changed during a signed effect"
                )


@dataclass(frozen=True, slots=True)
class _IssuedRequestStableView:
    step_id: int
    selection: RequestStateSelectionBatch
    view: StablePackedStateView


class RequestStateStableViewService(RequestStateRuntimeService[ModuleT]):
    """Bind stable physical views to staged logical selection ownership."""

    __slots__ = ("_views", "_issued")

    def __init__(
        self,
        core: _RequestStateRuntimeCore[ModuleT],
        resource: str,
        views: StablePackedStateViewArena,
    ) -> None:
        self._core = core
        self._resource = resource
        self._views = views
        self._issued: dict[
            StateViewCompletionFence,
            _IssuedRequestStableView,
        ] = {}

    def create_synchronous(
        self,
        step_id: int,
        selection: RequestStateSelectionBatch,
        tensors: tuple[torch.Tensor, ...],
    ) -> StablePackedStateView:
        with self._lock:
            self._require_staged_selection(step_id, selection)
            try:
                view = self._store.create_stable_view_synchronous(
                    self._views,
                    selection,
                    tensors,
                )
            except Exception:
                if self._views.poisoned:
                    self._store.fail_stop()
                raise
            self._track(step_id, selection, view)
            return view

    def create_cuda(
        self,
        step_id: int,
        selection: RequestStateSelectionBatch,
        tensors: tuple[torch.Tensor, ...],
        stream: torch.cuda.Stream,
    ) -> StablePackedStateView:
        with self._lock:
            self._require_staged_selection(step_id, selection)
            try:
                view = self._store.create_stable_view_cuda(
                    self._views,
                    selection,
                    tensors,
                    stream,
                )
            except Exception:
                if self._views.poisoned:
                    self._store.fail_stop()
                raise
            self._track(step_id, selection, view)
            return view

    def order_cuda(
        self,
        view: StablePackedStateView,
        stream: torch.cuda.Stream,
    ) -> None:
        with self._lock:
            issued = self._issued_view(view)
            self._require_staged_selection(
                issued.step_id,
                issued.selection,
            )
            self._views.order_cuda(view, stream)

    def complete_synchronous(self, view: StablePackedStateView) -> None:
        with self._lock:
            issued = self._issued_view(view)
            self._require_staged_selection(
                issued.step_id,
                issued.selection,
            )
            self._views.complete_synchronous(view.completion_fence)
            self._finish_issued(issued)

    def complete_cuda(
        self,
        view: StablePackedStateView,
        stream: torch.cuda.Stream,
    ) -> None:
        with self._lock:
            issued = self._issued_view(view)
            self._require_staged_selection(
                issued.step_id,
                issued.selection,
            )
            self._views.complete_cuda(view.completion_fence, stream)
            try:
                pin = self._store.pin_cuda(issued.selection, stream)
            except Exception:
                self._views.poison()
                self._store.fail_stop()
                raise
            self._completion_pins.setdefault(
                issued.step_id,
                [],
            ).append(pin)
            self._finish_issued(issued)

    def collect_completed(self) -> None:
        with self._lock:
            self._views.collect_completed()

    def wait_completion(self, completion: StateViewCompletionFence) -> None:
        with self._lock:
            self._views.wait_completion(completion)

    def _track(
        self,
        step_id: int,
        selection: RequestStateSelectionBatch,
        view: StablePackedStateView,
    ) -> None:
        completion = view.completion_fence
        if completion in self._issued:
            raise RuntimeError("stable view completion token is already issued")
        self._issued[completion] = _IssuedRequestStableView(
            step_id,
            selection,
            view,
        )
        if step_id in self._stable_view_counts:
            self._stable_view_counts[step_id] += 1
        else:
            self._stable_view_counts[step_id] = 1

    def _issued_view(
        self,
        view: StablePackedStateView,
    ) -> _IssuedRequestStableView:
        try:
            issued = self._issued[view.completion_fence]
        except KeyError as exc:
            raise RuntimeError("stable view is unknown or completed") from exc
        if issued.view is not view:
            raise RuntimeError("stable view differs from its issued token")
        return issued

    def _finish_issued(self, issued: _IssuedRequestStableView) -> None:
        self._issued.pop(issued.view.completion_fence)
        count = self._stable_view_counts[issued.step_id]
        if count == 1:
            self._stable_view_counts.pop(issued.step_id)
        else:
            self._stable_view_counts[issued.step_id] = count - 1


__all__ = [
    "IndexedRequestIdentityBatch",
    "IndexedRequestIdentityRow",
    "FixedBatchedSlotView",
    "RequestIdentityBatch",
    "RequestStateAcquisition",
    "RequestStateAsyncPin",
    "RequestStateEffectAudit",
    "RequestStateKey",
    "RequestStateLease",
    "RequestStateLossBatch",
    "RequestStateModuleGroup",
    "RequestStateSelected",
    "RequestStateRuntimeService",
    "RequestStateStableViewService",
    "RequestStateSelectionBatch",
    "RequestStateSelectionRow",
    "RequestStateVersion",
    "RequestTrainableStateStore",
    "ResidentSlotView",
    "StateEffectOperation",
    "StateEffectReceipt",
    "StateSlotEffect",
    "StateSlotRef",
]
