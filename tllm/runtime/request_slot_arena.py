"""Single logical authority for generational request-scoped slots."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

import torch

from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
    RequestKeyField,
)
from tllm.contracts.request_state import (
    RequestStateAcquisition,
    RequestStateAsyncPin,
    RequestStateKey,
    RequestStateLease,
    RequestStateSelectionBatch,
    RequestStateSelectionRow,
    RequestStateVersion,
    StateEffectOperation,
    StateEffectReceipt,
    StateSlotEffect,
    StateSlotRef,
)


@dataclass(frozen=True, slots=True)
class SlotAdmission:
    key: RequestStateKey
    slot: StateSlotRef


@dataclass(frozen=True, slots=True)
class RequestSlotAdmissionPlan:
    transaction_id: int
    epoch: int
    revision: int
    identities: tuple[RequestIdentity, ...]
    canonical_row_indices: tuple[int, ...]
    keys: tuple[RequestStateKey, ...]
    admissions: tuple[SlotAdmission, ...]
    planned_slot_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RequestSlotTransitionPlan:
    transaction_id: int
    epoch: int
    revision: int
    selection: RequestStateSelectionBatch
    transitions: tuple[StateSlotEffect, ...]


@dataclass(frozen=True, slots=True)
class RequestSlotPinPlan:
    transaction_id: int
    epoch: int
    revision: int
    selection: RequestStateSelectionBatch
    pin: RequestStateAsyncPin


@dataclass(slots=True)
class _FreeSlot:
    generation: int = 0
    version: int = 0
    pending_receipts: int = 0


@dataclass(slots=True)
class _BoundSlot:
    generation: int
    version: int
    key: RequestStateKey
    aliases: set[RequestIdentity] = field(default_factory=set[RequestIdentity])
    selection_leases: int = 0
    pins: int = 0
    pending_receipts: int = 0


_Slot = _FreeSlot | _BoundSlot


@dataclass(frozen=True, slots=True)
class _ActiveSelection:
    rows: tuple[RequestStateSelectionRow, ...]
    leases: tuple[RequestStateLease, ...]
    before_versions: tuple[int, ...]
    slot_ids: tuple[int, ...]
    lease_slot_ids: tuple[int, ...]
    device_slot_ids: torch.Tensor
    device_lease_slot_ids: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ActivePin:
    epoch: int
    slots: tuple[StateSlotRef, ...]


@dataclass(frozen=True, slots=True)
class _NoPendingTransaction:
    pass


@dataclass(frozen=True, slots=True)
class _PendingAdmission:
    plan: RequestSlotAdmissionPlan
    device_slot_ids: torch.Tensor
    selection_slot_ids: torch.Tensor
    lease_slot_ids: tuple[int, ...]
    device_lease_slot_ids: torch.Tensor
    selection_lease_slot_ids: torch.Tensor
    valid_mask: torch.Tensor
    selection_valid_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class _PendingTransition:
    plan: RequestSlotTransitionPlan


@dataclass(frozen=True, slots=True)
class _PendingPin:
    plan: RequestSlotPinPlan


_PendingTransaction = (
    _NoPendingTransaction
    | _PendingAdmission
    | _PendingTransition
    | _PendingPin
)
_NO_PENDING = _NoPendingTransaction()


class RequestSlotArena:
    """Own keys, aliases, generations, versions, leases, and receipts.

    A physical adapter receives a reserved plan and completes its fallible work
    before logical commit. While a plan is pending, every operation that could
    change its preconditions is closed. Aborted preparation publishes nothing;
    a physical commit failure poisons the arena and therefore fails stop.
    """

    __slots__ = (
        "_resource",
        "_key_fields",
        "_capacity",
        "_epoch",
        "_device",
        "_slots",
        "_slot_by_key",
        "_key_by_identity",
        "_active_selections",
        "_active_pins",
        "_issued_receipts",
        "_pending",
        "_next_selection_id",
        "_next_pin_id",
        "_next_receipt_id",
        "_next_transaction_id",
        "_revision",
        "_poisoned",
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
    ) -> None:
        if not resource:
            raise ValueError("request slot arena requires a resource")
        if not key_fields or len(key_fields) != len(set(key_fields)):
            raise ValueError("request slot arena requires unique key fields")
        if capacity < 1 or epoch < 1:
            raise ValueError("request slot arena requires positive capacity and epoch")
        self._resource = resource
        self._key_fields = key_fields
        self._capacity = capacity
        self._epoch = epoch
        self._device = device
        self._slots: list[_Slot] = [_FreeSlot() for _ in range(capacity)]
        self._slot_by_key: dict[RequestStateKey, int] = {}
        self._key_by_identity: dict[RequestIdentity, RequestStateKey] = {}
        self._active_selections: dict[int, _ActiveSelection] = {}
        self._active_pins: dict[int, _ActivePin] = {}
        self._issued_receipts: dict[int, StateEffectReceipt] = {}
        self._pending: _PendingTransaction = _NO_PENDING
        self._next_selection_id = 1
        self._next_pin_id = 1
        self._next_receipt_id = 1
        self._next_transaction_id = 1
        self._revision = 0
        self._poisoned = False
        self._lock = RLock()

    @property
    def resource(self) -> str:
        return self._resource

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def device(self) -> torch.device:
        return self._device

    def poison(self) -> None:
        with self._lock:
            self._poisoned = True

    def plan_admission(
        self,
        identities: tuple[RequestIdentity, ...],
        canonical_row_indices: tuple[int, ...],
    ) -> RequestSlotAdmissionPlan:
        if not identities or len(identities) != len(canonical_row_indices):
            raise ValueError("request slot admission rows must be non-empty and aligned")
        if canonical_row_indices != tuple(sorted(set(canonical_row_indices))):
            raise ValueError("request slot admission rows must be strictly increasing")
        keys = tuple(
            RequestStateKey.project(identity, self._key_fields)
            for identity in identities
        )
        unique_keys = tuple(dict.fromkeys(keys))
        with self._lock:
            self._require_live()
            self._require_no_pending_plan()
            for identity, key in zip(identities, keys, strict=True):
                if (
                    identity in self._key_by_identity
                    and self._key_by_identity[identity] != key
                ):
                    raise RuntimeError("request identity changed its signed state key")
            new_keys = tuple(
                key for key in unique_keys if key not in self._slot_by_key
            )
            free_slots = tuple(
                slot_id
                for slot_id, slot in enumerate(self._slots)
                if self._slot_is_available(slot)
            )
            if len(new_keys) > len(free_slots):
                raise RuntimeError(
                    "request slot capacity exhausted: "
                    f"required={len(new_keys)}, available={len(free_slots)}"
                )
            selected_free_slots = free_slots[: len(new_keys)]
            planned = dict(self._slot_by_key)
            admissions: list[SlotAdmission] = []
            for key, slot_id in zip(
                new_keys,
                selected_free_slots,
                strict=True,
            ):
                match self._slots[slot_id]:
                    case _FreeSlot() as free:
                        slot_ref = StateSlotRef(slot_id, free.generation + 1)
                    case _:
                        raise AssertionError("planned request slot is not free")
                planned[key] = slot_id
                admissions.append(SlotAdmission(key, slot_ref))
            planned_slot_ids = tuple(planned[key] for key in keys)
            device_slot_ids = torch.tensor(
                planned_slot_ids,
                dtype=torch.long,
                device=self._device,
            )
            lease_slot_ids = tuple(planned[key] for key in unique_keys)
            device_lease_slot_ids = torch.tensor(
                lease_slot_ids,
                dtype=torch.long,
                device=self._device,
            )
            valid_mask = torch.ones(
                (len(planned_slot_ids),),
                dtype=torch.bool,
                device=self._device,
            )
            plan = RequestSlotAdmissionPlan(
                transaction_id=self._next_transaction_id,
                epoch=self._epoch,
                revision=self._revision,
                identities=identities,
                canonical_row_indices=canonical_row_indices,
                keys=keys,
                admissions=tuple(admissions),
                planned_slot_ids=planned_slot_ids,
            )
            self._pending = _PendingAdmission(
                plan,
                device_slot_ids,
                device_slot_ids.clone(),
                lease_slot_ids,
                device_lease_slot_ids,
                device_lease_slot_ids.clone(),
                valid_mask,
                valid_mask.clone(),
            )
            self._next_transaction_id += 1
            return plan

    def abort_admission(
        self, plan: RequestSlotAdmissionPlan, *, poison: bool
    ) -> None:
        with self._lock:
            self._require_pending_admission(plan)
            self._pending = _NO_PENDING
            if poison:
                self._poisoned = True

    def commit_admission(
        self, plan: RequestSlotAdmissionPlan
    ) -> RequestStateAcquisition:
        with self._lock:
            pending = self._require_pending_admission(plan)
            admission_by_key = {
                admission.key: admission for admission in plan.admissions
            }
            lease_by_key: dict[RequestStateKey, RequestStateLease] = {}
            for key in dict.fromkeys(plan.keys):
                if key in admission_by_key:
                    slot_ref = admission_by_key[key].slot
                else:
                    slot_id = self._slot_by_key[key]
                    bound = self._bound_slot(slot_id)
                    slot_ref = StateSlotRef(slot_id, bound.generation)
                lease_by_key[key] = RequestStateLease(
                    key=key,
                    slot=slot_ref,
                    epoch=self._epoch,
                )
            rows = tuple(
                RequestStateSelectionRow(identity, lease_by_key[key], row_index)
                for identity, key, row_index in zip(
                    plan.identities,
                    plan.keys,
                    plan.canonical_row_indices,
                    strict=True,
                )
            )
            leases = tuple(lease_by_key[key] for key in dict.fromkeys(plan.keys))
            before_versions = tuple(
                0
                if lease.key in admission_by_key
                else self._bound_slot(lease.slot.slot_id).version
                for lease in leases
            )
            selection_id = self._next_selection_id
            active = _ActiveSelection(
                rows=rows,
                leases=leases,
                before_versions=before_versions,
                slot_ids=plan.planned_slot_ids,
                lease_slot_ids=pending.lease_slot_ids,
                device_slot_ids=pending.device_slot_ids,
                device_lease_slot_ids=pending.device_lease_slot_ids,
                valid_mask=pending.valid_mask,
            )
            selection = RequestStateSelectionBatch(
                resource=self._resource,
                selection_id=selection_id,
                epoch=self._epoch,
                rows=rows,
                leases=leases,
                before_versions=before_versions,
                device_slot_ids=pending.selection_slot_ids,
                device_lease_slot_ids=pending.selection_lease_slot_ids,
                valid_mask=pending.selection_valid_mask,
            )
            effects = tuple(
                StateSlotEffect(
                    admission.key,
                    admission.slot,
                    0,
                    0,
                )
                for admission in plan.admissions
            )
            receipt = self._new_receipt(
                StateEffectOperation.ADMIT,
                selection_id,
                effects,
            )
            for admission in plan.admissions:
                slot_id = admission.slot.slot_id
                match self._slots[slot_id]:
                    case _FreeSlot(generation=generation, pending_receipts=0) if (
                        admission.slot.generation == generation + 1
                    ):
                        pass
                    case _:
                        raise AssertionError(
                            "reserved request slot changed before commit"
                        )
            for admission in plan.admissions:
                self._slots[admission.slot.slot_id] = _BoundSlot(
                    generation=admission.slot.generation,
                    version=0,
                    key=admission.key,
                )
                self._slot_by_key[admission.key] = admission.slot.slot_id
            for identity, key in zip(plan.identities, plan.keys, strict=True):
                self._key_by_identity[identity] = key
                self._bound_slot(self._slot_by_key[key]).aliases.add(identity)
            self._active_selections[selection_id] = active
            for lease in leases:
                self._bound_slot(lease.slot.slot_id).selection_leases += 1
            self._pending = _NO_PENDING
            self._next_selection_id += 1
            self._revision += 1
            self._publish_receipt(receipt)
            return RequestStateAcquisition(selection, receipt)

    def validate_selection(
        self, selection: RequestStateSelectionBatch, *, audited: bool
    ) -> None:
        with self._lock:
            self._require_live()
            self._require_no_pending_plan()
            active = self._active_selection(selection)
            if audited and any(
                self._bound_slot(lease.slot.slot_id).pending_receipts
                for lease in active.leases
            ):
                raise RuntimeError("request slot selection has an unaudited effect")

    def selection_versions(
        self, selection: RequestStateSelectionBatch
    ) -> tuple[RequestStateVersion, ...]:
        with self._lock:
            self.validate_selection(selection, audited=True)
            return tuple(
                RequestStateVersion(
                    lease.key,
                    lease.slot,
                    self._bound_slot(lease.slot.slot_id).version,
                )
                for lease in selection.leases
            )

    def read_receipt(
        self, selection: RequestStateSelectionBatch
    ) -> StateEffectReceipt:
        with self._lock:
            self.validate_selection(selection, audited=True)
            effects = tuple(
                StateSlotEffect(
                    lease.key,
                    lease.slot,
                    self._bound_slot(lease.slot.slot_id).version,
                    self._bound_slot(lease.slot.slot_id).version,
                )
                for lease in selection.leases
            )
            receipt = self._new_receipt(
                StateEffectOperation.READ,
                selection.selection_id,
                effects,
            )
            self._require_receipt_preconditions(receipt)
            self._publish_receipt(receipt)
            return receipt

    def plan_transition(
        self, selection: RequestStateSelectionBatch
    ) -> RequestSlotTransitionPlan:
        with self._lock:
            self.validate_selection(selection, audited=True)
            plan = RequestSlotTransitionPlan(
                transaction_id=self._next_transaction_id,
                epoch=self._epoch,
                revision=self._revision,
                selection=selection,
                transitions=tuple(
                    StateSlotEffect(
                        lease.key,
                        lease.slot,
                        self._bound_slot(lease.slot.slot_id).version,
                        self._bound_slot(lease.slot.slot_id).version + 1,
                    )
                    for lease in selection.leases
                ),
            )
            self._pending = _PendingTransition(plan)
            self._next_transaction_id += 1
            return plan

    def abort_transition(
        self, plan: RequestSlotTransitionPlan, *, poison: bool
    ) -> None:
        with self._lock:
            self._require_pending_transition(plan)
            self._pending = _NO_PENDING
            if poison:
                self._poisoned = True

    def commit_transition(
        self, plan: RequestSlotTransitionPlan
    ) -> StateEffectReceipt:
        with self._lock:
            self._require_pending_transition(plan)
            receipt = self._new_receipt(
                StateEffectOperation.TRANSITION,
                plan.selection.selection_id,
                plan.transitions,
            )
            for effect in plan.transitions:
                bound = self._bound_slot(effect.slot.slot_id)
                if (
                    bound.generation != effect.slot.generation
                    or bound.key != effect.key
                    or bound.version != effect.before_version
                    or bound.pending_receipts
                    or effect.after_version != effect.before_version + 1
                ):
                    raise AssertionError("request slot transition is not monotonic")
            for effect in plan.transitions:
                self._bound_slot(effect.slot.slot_id).version = effect.after_version
            self._pending = _NO_PENDING
            self._revision += 1
            self._publish_receipt(receipt)
            return receipt

    def release_selection(self, selection: RequestStateSelectionBatch) -> None:
        with self._lock:
            self.validate_selection(selection, audited=True)
            active = self._active_selection(selection)
            for lease in active.leases:
                self._bound_slot(lease.slot.slot_id).selection_leases -= 1
            self._active_selections.pop(selection.selection_id)

    def plan_pin(
        self, selection: RequestStateSelectionBatch
    ) -> RequestSlotPinPlan:
        with self._lock:
            self.validate_selection(selection, audited=True)
            slots = tuple(lease.slot for lease in selection.leases)
            pin = RequestStateAsyncPin(
                self._resource,
                self._next_pin_id,
                self._epoch,
                slots,
            )
            plan = RequestSlotPinPlan(
                self._next_transaction_id,
                self._epoch,
                self._revision,
                selection,
                pin,
            )
            self._pending = _PendingPin(plan)
            self._next_pin_id += 1
            self._next_transaction_id += 1
            return plan

    def abort_pin(self, plan: RequestSlotPinPlan, *, poison: bool) -> None:
        with self._lock:
            self._require_pending_pin(plan)
            self._pending = _NO_PENDING
            if poison:
                self._poisoned = True

    def commit_pin(self, plan: RequestSlotPinPlan) -> RequestStateAsyncPin:
        with self._lock:
            self._require_pending_pin(plan)
            active = _ActivePin(self._epoch, plan.pin.slots)
            self._active_pins[plan.pin.pin_id] = active
            for slot in plan.pin.slots:
                self._bound_slot(slot.slot_id).pins += 1
            self._pending = _NO_PENDING
            self._revision += 1
            return plan.pin

    def validate_pin(self, pin: RequestStateAsyncPin) -> None:
        with self._lock:
            self._require_live()
            self._require_no_pending_plan()
            active = self._active_pin(pin)
            for slot in active.slots:
                bound = self._bound_slot(slot.slot_id)
                if bound.generation != slot.generation:
                    raise RuntimeError("request slot pin contains a stale generation")

    def release_pin(self, pin: RequestStateAsyncPin) -> None:
        with self._lock:
            self.validate_pin(pin)
            active = self._active_pin(pin)
            for slot in active.slots:
                self._bound_slot(slot.slot_id).pins -= 1
            self._active_pins.pop(pin.pin_id)
            self._revision += 1

    def retire(self, batch: RequestFinishedBatch) -> StateEffectReceipt:
        if len(batch.requests) != len(set(batch.requests)):
            raise ValueError("request-finished batch contains duplicate identities")
        with self._lock:
            self._require_live()
            self._require_no_pending_plan()
            matched = tuple(
                identity
                for identity in batch.requests
                if identity in self._key_by_identity
            )
            retiring_by_key: dict[RequestStateKey, set[RequestIdentity]] = {}
            for identity in matched:
                key = self._key_by_identity[identity]
                retiring_by_key.setdefault(key, set()).add(identity)
            released_keys: list[RequestStateKey] = []
            for key, retiring in retiring_by_key.items():
                bound = self._bound_slot(self._slot_by_key[key])
                if not bound.aliases.difference(retiring):
                    if (
                        bound.selection_leases
                        or bound.pins
                        or bound.pending_receipts
                    ):
                        raise RuntimeError(
                            "cannot retire request state with active leases, "
                            "pins, or receipts"
                        )
                    released_keys.append(key)
            effects = tuple(
                StateSlotEffect(
                    key,
                    StateSlotRef(
                        self._slot_by_key[key],
                        self._bound_slot(self._slot_by_key[key]).generation,
                    ),
                    self._bound_slot(self._slot_by_key[key]).version,
                    self._bound_slot(self._slot_by_key[key]).version,
                )
                for key in released_keys
            )
            receipt = self._new_receipt(
                StateEffectOperation.RETIRE,
                0,
                effects,
            )
            for effect in effects:
                bound = self._bound_slot(effect.slot.slot_id)
                if (
                    bound.generation != effect.slot.generation
                    or bound.key != effect.key
                    or bound.version != effect.after_version
                    or bound.pending_receipts
                ):
                    raise AssertionError(
                        "request slot retirement changed before commit"
                    )
            for identity in matched:
                key = self._key_by_identity.pop(identity)
                self._bound_slot(self._slot_by_key[key]).aliases.remove(identity)
            for key in released_keys:
                slot_id = self._slot_by_key.pop(key)
                bound = self._bound_slot(slot_id)
                self._slots[slot_id] = _FreeSlot(
                    generation=bound.generation,
                    version=bound.version,
                )
            self._revision += 1
            self._publish_receipt(receipt)
            return receipt

    def versions(self) -> tuple[RequestStateVersion, ...]:
        with self._lock:
            self._require_live()
            return tuple(
                RequestStateVersion(
                    key,
                    StateSlotRef(slot_id, self._bound_slot(slot_id).generation),
                    self._bound_slot(slot_id).version,
                )
                for key, slot_id in sorted(
                    self._slot_by_key.items(), key=lambda item: item[1]
                )
            )

    def replace_epoch(self, epoch: int) -> None:
        with self._lock:
            self._validate_epoch_replacement(epoch)
            self._epoch = epoch
            self._revision += 1

    def validate_epoch_replacement(self, epoch: int) -> None:
        with self._lock:
            self._validate_epoch_replacement(epoch)

    def _validate_epoch_replacement(self, epoch: int) -> None:
        if epoch <= self._epoch:
            raise ValueError("request slot epoch replacement must advance")
        self._require_live()
        self._require_no_pending_plan()
        if (
            self._slot_by_key
            or self._active_selections
            or self._active_pins
            or self._issued_receipts
        ):
            raise RuntimeError(
                "cannot replace request state epoch while requests are active"
            )

    def consume_receipt(self, receipt: StateEffectReceipt) -> None:
        with self._lock:
            self._require_live()
            self._require_no_pending_plan()
            try:
                issued = self._issued_receipts[receipt.receipt_id]
            except KeyError as exc:
                raise RuntimeError("request slot receipt is unknown or consumed") from exc
            if issued != receipt:
                raise RuntimeError("request slot receipt differs from issued effect")
            self._require_receipt_preconditions(receipt, pending_receipts=1)
            for slot_ref in dict.fromkeys(effect.slot for effect in receipt.slots):
                self._slots[slot_ref.slot_id].pending_receipts -= 1
            self._issued_receipts.pop(receipt.receipt_id)

    def _active_selection(
        self, selection: RequestStateSelectionBatch
    ) -> _ActiveSelection:
        if selection.resource != self._resource or selection.epoch != self._epoch:
            raise RuntimeError("request slot selection belongs to another epoch")
        try:
            active = self._active_selections[selection.selection_id]
        except KeyError as exc:
            raise RuntimeError(
                "request state selection is no longer active"
            ) from exc
        if (
            selection.rows != active.rows
            or selection.leases != active.leases
            or selection.before_versions != active.before_versions
        ):
            raise RuntimeError(
                "request state selection leases differ from admission"
            )
        if (
            selection.device_slot_ids.dtype is not torch.long
            or selection.device_slot_ids.device != self._device
            or tuple(selection.device_slot_ids.shape) != (len(active.slot_ids),)
            or not torch.equal(selection.device_slot_ids, active.device_slot_ids)
            or selection.device_lease_slot_ids.dtype is not torch.long
            or selection.device_lease_slot_ids.device != self._device
            or tuple(selection.device_lease_slot_ids.shape)
            != (len(active.lease_slot_ids),)
            or not torch.equal(
                selection.device_lease_slot_ids,
                active.device_lease_slot_ids,
            )
            or selection.valid_mask.dtype is not torch.bool
            or selection.valid_mask.device != self._device
            or tuple(selection.valid_mask.shape) != (len(active.slot_ids),)
            or not torch.equal(selection.valid_mask, active.valid_mask)
        ):
            raise RuntimeError(
                "request state selection tensors differ from admission"
            )
        for lease in active.leases:
            bound = self._bound_slot(lease.slot.slot_id)
            if (
                bound.generation != lease.slot.generation
                or bound.key != lease.key
                or lease.epoch != self._epoch
            ):
                raise RuntimeError("request slot selection contains a stale lease")
        return active

    def _active_pin(self, pin: RequestStateAsyncPin) -> _ActivePin:
        try:
            active = self._active_pins[pin.pin_id]
        except KeyError as exc:
            raise RuntimeError("request slot pin is unknown") from exc
        if (
            pin.resource != self._resource
            or pin.epoch != self._epoch
            or active.epoch != pin.epoch
            or active.slots != pin.slots
        ):
            raise RuntimeError("request slot pin differs from its lease")
        return active

    def _new_receipt(
        self,
        operation: StateEffectOperation,
        selection_id: int,
        effects: tuple[StateSlotEffect, ...],
    ) -> StateEffectReceipt:
        return StateEffectReceipt(
            receipt_id=self._next_receipt_id,
            resource=self._resource,
            operation=operation,
            selection_id=selection_id,
            epoch=self._epoch,
            slots=effects,
        )

    def _require_receipt_preconditions(
        self,
        receipt: StateEffectReceipt,
        *,
        pending_receipts: int = 0,
    ) -> None:
        for effect in receipt.slots:
            slot = self._slots[effect.slot.slot_id]
            if (
                slot.generation != effect.slot.generation
                or slot.version != effect.after_version
                or slot.pending_receipts != pending_receipts
            ):
                raise RuntimeError("request slot receipt no longer matches state")
            if receipt.operation is not StateEffectOperation.RETIRE:
                bound = self._bound_slot(effect.slot.slot_id)
                if bound.key != effect.key:
                    raise RuntimeError("request slot receipt key changed")

    def _publish_receipt(self, receipt: StateEffectReceipt) -> None:
        slots = tuple(dict.fromkeys(effect.slot for effect in receipt.slots))
        self._issued_receipts[receipt.receipt_id] = receipt
        self._next_receipt_id += 1
        for slot_ref in slots:
            self._slots[slot_ref.slot_id].pending_receipts += 1

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError("request slot arena is poisoned")

    def _require_pending_admission(
        self, plan: RequestSlotAdmissionPlan
    ) -> _PendingAdmission:
        self._require_live()
        match self._pending:
            case _PendingAdmission(plan=issued) as pending if (
                issued == plan
                and plan.epoch == self._epoch
                and plan.revision == self._revision
            ):
                return pending
            case _:
                raise RuntimeError("request slot admission plan is stale")

    def _require_pending_transition(
        self, plan: RequestSlotTransitionPlan
    ) -> None:
        self._require_live()
        match self._pending:
            case _PendingTransition(plan=issued) if (
                issued == plan
                and plan.epoch == self._epoch
                and plan.revision == self._revision
            ):
                return
            case _:
                raise RuntimeError("request slot transition plan is stale")

    def _require_pending_pin(self, plan: RequestSlotPinPlan) -> None:
        self._require_live()
        match self._pending:
            case _PendingPin(plan=issued) if (
                issued == plan
                and plan.epoch == self._epoch
                and plan.revision == self._revision
            ):
                return
            case _:
                raise RuntimeError("request slot pin plan is stale")

    def _bound_slot(self, slot_id: int) -> _BoundSlot:
        match self._slots[slot_id]:
            case _BoundSlot() as bound:
                return bound
            case _:
                raise RuntimeError("request slot is not bound")

    def _require_no_pending_plan(self) -> None:
        match self._pending:
            case _NoPendingTransaction():
                return
            case _:
                raise RuntimeError("request slot arena already has a pending plan")

    @staticmethod
    def _slot_is_available(slot: _Slot) -> bool:
        match slot:
            case _FreeSlot(pending_receipts=0):
                return True
            case _:
                return False

__all__ = [
    "RequestSlotAdmissionPlan",
    "RequestSlotArena",
    "RequestSlotPinPlan",
    "RequestSlotTransitionPlan",
    "SlotAdmission",
]
