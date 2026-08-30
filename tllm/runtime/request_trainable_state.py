"""Fixed-capacity request-scoped trainable state with typed leases.

The store owns logical request aliases, physical slot generations, modules,
optimizers, and version transitions.  It does not choose execution order or
placement; a signed Region supplies the selection whose effects are audited.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from typing import Generic, TypeVar

import torch

from tllm.contracts.request_identity import (
    RequestFinishedBatch,
    RequestIdentity,
    RequestKeyField,
)


ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


@dataclass(frozen=True, slots=True)
class RequestStateKey:
    """Canonical projection of one identity under a signed key declaration."""

    fields: tuple[RequestKeyField, ...]
    request_id: str
    prompt_idx: int
    sample_idx: int

    @classmethod
    def project(
        cls,
        identity: RequestIdentity,
        fields: tuple[RequestKeyField, ...],
    ) -> "RequestStateKey":
        if not fields:
            raise ValueError("request state key projection requires fields")
        if len(fields) != len(set(fields)):
            raise ValueError("request state key fields must be unique")
        if RequestKeyField.REQUEST_ID in fields and not identity.request_id:
            raise ValueError("request_id state key requires a non-empty request id")
        if RequestKeyField.PROMPT_IDX in fields and identity.prompt_idx < 0:
            raise ValueError("prompt_idx state key requires resolved prompt metadata")
        if RequestKeyField.SAMPLE_IDX in fields and identity.sample_idx < 0:
            raise ValueError("sample_idx state key requires resolved sample metadata")
        return cls(
            fields=fields,
            request_id=(
                identity.request_id
                if RequestKeyField.REQUEST_ID in fields
                else ""
            ),
            prompt_idx=(
                identity.prompt_idx
                if RequestKeyField.PROMPT_IDX in fields
                else -1
            ),
            sample_idx=(
                identity.sample_idx
                if RequestKeyField.SAMPLE_IDX in fields
                else -1
            ),
        )

    def wire_key(self) -> str:
        """Closed control-plane representation; no runtime type inspection."""

        parts: list[str] = []
        for field in self.fields:
            match field:
                case RequestKeyField.REQUEST_ID:
                    parts.append(f"request_id={self.request_id}")
                case RequestKeyField.PROMPT_IDX:
                    parts.append(f"prompt_idx={self.prompt_idx}")
                case RequestKeyField.SAMPLE_IDX:
                    parts.append(f"sample_idx={self.sample_idx}")
        return "|".join(parts)


@dataclass(frozen=True, slots=True, order=True)
class StateSlotRef:
    slot_id: int
    generation: int

    def __post_init__(self) -> None:
        if self.slot_id < 0:
            raise ValueError("request state slot id must be non-negative")
        if self.generation < 1:
            raise ValueError("request state slot generation must be positive")

    def wire_key(self) -> str:
        return f"{self.slot_id}:{self.generation}"


@dataclass(frozen=True, slots=True)
class RequestStateLease:
    key: RequestStateKey
    slot: StateSlotRef
    epoch: int


@dataclass(frozen=True, slots=True)
class RequestStateSelectionRow:
    identity: RequestIdentity
    lease: RequestStateLease


@dataclass(frozen=True, slots=True)
class RequestStateSelectionBatch:
    resource: str
    selection_id: int
    epoch: int
    rows: tuple[RequestStateSelectionRow, ...]
    leases: tuple[RequestStateLease, ...]
    device_slot_ids: torch.Tensor
    valid_mask: torch.Tensor


class StateEffectOperation(StrEnum):
    ADMIT = "admit"
    READ = "read"
    TRANSITION = "transition"
    RETIRE = "retire"


@dataclass(frozen=True, slots=True)
class StateSlotEffect:
    key: RequestStateKey
    slot: StateSlotRef
    before_version: int
    after_version: int


@dataclass(frozen=True, slots=True)
class StateEffectReceipt:
    receipt_id: int
    resource: str
    operation: StateEffectOperation
    selection_id: int
    epoch: int
    slots: tuple[StateSlotEffect, ...]


@dataclass(frozen=True, slots=True)
class RequestStateAcquisition:
    selection: RequestStateSelectionBatch
    admission: StateEffectReceipt


@dataclass(frozen=True, slots=True)
class RequestStateVersion:
    key: RequestStateKey
    slot: StateSlotRef
    version: int


@dataclass(frozen=True, slots=True)
class RequestStateModuleGroup(Generic[ModuleT]):
    lease: RequestStateLease
    row_indices: torch.Tensor
    module: ModuleT


@dataclass(frozen=True, slots=True)
class RequestStateAsyncPin:
    resource: str
    pin_id: int
    epoch: int
    slots: tuple[StateSlotRef, ...]


@dataclass(slots=True)
class _RequestStateSlot(Generic[ModuleT]):
    module: ModuleT
    optimizer: torch.optim.Optimizer
    generation: int = 0
    version: int = 0
    key: RequestStateKey | None = None
    aliases: set[RequestIdentity] = field(
        default_factory=set[RequestIdentity]
    )
    selection_leases: int = 0
    async_pins: int = 0
    pending_receipts: int = 0


@dataclass(frozen=True, slots=True)
class _ActiveSelection:
    rows: tuple[RequestStateSelectionRow, ...]
    leases: tuple[RequestStateLease, ...]
    slot_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ActivePin:
    slots: tuple[StateSlotRef, ...]
    completion: torch.cuda.Event | None


class RequestTrainableStateStore(Generic[ModuleT]):
    """A fixed slot bank with generation-safe request ownership.

    Modules are allocated during cold specialization and keep their parameter
    addresses for the store lifetime.  Optimizers are intentionally eager in
    this first capability: slot reuse reconstructs optimizer state, so Rust
    must not classify request-scoped transitions as capture-compatible yet.
    """

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
        if not resource:
            raise ValueError("request trainable state requires a resource")
        if not key_fields or len(key_fields) != len(set(key_fields)):
            raise ValueError("request trainable state requires unique key fields")
        if capacity < 1:
            raise ValueError("request trainable state capacity must be positive")
        if epoch < 1:
            raise ValueError("request trainable state epoch must be positive")

        modules = tuple(module_factory() for _slot_id in range(capacity))
        if len({id(module) for module in modules}) != capacity:
            raise RuntimeError("request state module slots must be distinct")
        canonical_parameters = tuple(
            parameter.detach().clone() for parameter in modules[0].parameters()
        )
        canonical_buffers = tuple(
            buffer.detach().clone() for buffer in modules[0].buffers()
        )
        if not canonical_parameters:
            raise ValueError("request trainable state modules require parameters")
        parameter_ids: set[int] = set()
        parameter_addresses: set[int] = set()
        buffer_ids: set[int] = set()
        buffer_addresses: set[int] = set()
        for module in modules:
            parameters = tuple(module.parameters())
            buffers = tuple(module.buffers())
            if len(parameters) != len(canonical_parameters):
                raise RuntimeError("request state module parameter layouts differ")
            if len(buffers) != len(canonical_buffers):
                raise RuntimeError("request state module buffer layouts differ")
            if any(parameter.numel() < 1 for parameter in parameters):
                raise RuntimeError("request state module parameters must be non-empty")
            if any(
                tuple(parameter.shape) != tuple(initial.shape)
                or parameter.dtype is not initial.dtype
                or parameter.device != initial.device
                for parameter, initial in zip(
                    parameters, canonical_parameters, strict=True
                )
            ):
                raise RuntimeError("request state module parameter layouts differ")
            if any(
                tuple(buffer.shape) != tuple(initial.shape)
                or buffer.dtype is not initial.dtype
                or buffer.device != initial.device
                for buffer, initial in zip(
                    buffers, canonical_buffers, strict=True
                )
            ):
                raise RuntimeError("request state module buffer layouts differ")
            module_parameter_ids = {id(parameter) for parameter in parameters}
            module_parameter_addresses = {
                parameter.data_ptr() for parameter in parameters
            }
            module_buffer_ids = {id(buffer) for buffer in buffers}
            module_buffer_addresses = {
                buffer.data_ptr() for buffer in buffers if buffer.numel()
            }
            if (
                parameter_ids.intersection(module_parameter_ids)
                or parameter_addresses.intersection(module_parameter_addresses)
                or buffer_ids.intersection(module_buffer_ids)
                or buffer_addresses.intersection(module_buffer_addresses)
            ):
                raise RuntimeError(
                    "request state module slots must not share tensor storage"
                )
            parameter_ids.update(module_parameter_ids)
            parameter_addresses.update(module_parameter_addresses)
            buffer_ids.update(module_buffer_ids)
            buffer_addresses.update(module_buffer_addresses)
            with torch.no_grad():
                for parameter, initial in zip(
                    parameters, canonical_parameters, strict=True
                ):
                    parameter.copy_(initial)
                for buffer, initial in zip(
                    buffers, canonical_buffers, strict=True
                ):
                    buffer.copy_(initial)

        self._resource = resource
        self._key_fields = key_fields
        self._capacity = capacity
        self._epoch = epoch
        self._device = device
        self._optimizer_factory = optimizer_factory
        self._initial_parameters = canonical_parameters
        self._initial_buffers = canonical_buffers
        self._slots = [
            _RequestStateSlot(module, optimizer_factory(module))
            for module in modules
        ]
        self._slot_by_key: dict[RequestStateKey, int] = {}
        self._key_by_identity: dict[RequestIdentity, RequestStateKey] = {}
        self._active_selections: dict[int, _ActiveSelection] = {}
        self._active_pins: dict[int, _ActivePin] = {}
        self._issued_receipts: dict[int, StateEffectReceipt] = {}
        self._next_selection_id = 1
        self._next_pin_id = 1
        self._next_receipt_id = 1
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

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError("request trainable state store is poisoned")

    def _issue_receipt(
        self,
        operation: StateEffectOperation,
        selection_id: int,
        slots: tuple[StateSlotEffect, ...],
    ) -> StateEffectReceipt:
        unique_slots = tuple(dict.fromkeys(effect.slot for effect in slots))
        for slot_ref in unique_slots:
            slot = self._slots[slot_ref.slot_id]
            if slot.generation != slot_ref.generation:
                raise RuntimeError("request state effect references a stale slot")
            if slot.pending_receipts:
                raise RuntimeError("request state slot has an unaudited effect")
        receipt = StateEffectReceipt(
            receipt_id=self._next_receipt_id,
            resource=self._resource,
            operation=operation,
            selection_id=selection_id,
            epoch=self._epoch,
            slots=slots,
        )
        self._next_receipt_id += 1
        self._issued_receipts[receipt.receipt_id] = receipt
        for slot_ref in unique_slots:
            self._slots[slot_ref.slot_id].pending_receipts += 1
        return receipt

    def consume_receipt(self, receipt: StateEffectReceipt) -> None:
        """Acknowledge one exact store-issued effect after version audit."""

        with self._lock:
            try:
                issued = self._issued_receipts[receipt.receipt_id]
            except KeyError as exc:
                raise RuntimeError("request state receipt is unknown or consumed") from exc
            if issued != receipt:
                raise RuntimeError("request state receipt differs from issued effect")
            for effect in receipt.slots:
                slot = self._slots[effect.slot.slot_id]
                if (
                    slot.generation != effect.slot.generation
                    or slot.version != effect.after_version
                ):
                    raise RuntimeError(
                        "request state receipt no longer matches its slot version"
                    )
                if (
                    receipt.operation is not StateEffectOperation.RETIRE
                    and slot.key != effect.key
                ):
                    raise RuntimeError(
                        "request state receipt no longer matches its logical key"
                    )
            for slot_ref in dict.fromkeys(
                effect.slot for effect in receipt.slots
            ):
                self._slots[slot_ref.slot_id].pending_receipts -= 1
            self._issued_receipts.pop(receipt.receipt_id)

    def _validate_slot_layout(self, slot_id: int) -> None:
        slot = self._slots[slot_id]
        parameters = tuple(slot.module.parameters())
        buffers = tuple(slot.module.buffers())
        if len(parameters) != len(self._initial_parameters) or any(
            tuple(parameter.shape) != tuple(initial.shape)
            or parameter.dtype is not initial.dtype
            or parameter.device != initial.device
            for parameter, initial in zip(
                parameters, self._initial_parameters, strict=True
            )
        ):
            raise RuntimeError("request state slot parameter layout changed")
        if len(buffers) != len(self._initial_buffers) or any(
            tuple(buffer.shape) != tuple(initial.shape)
            or buffer.dtype is not initial.dtype
            or buffer.device != initial.device
            for buffer, initial in zip(
                buffers, self._initial_buffers, strict=True
            )
        ):
            raise RuntimeError("request state slot buffer layout changed")

    def _reset_slot(
        self,
        slot_id: int,
        key: RequestStateKey,
        optimizer: torch.optim.Optimizer,
    ) -> StateSlotRef:
        slot = self._slots[slot_id]
        if (
            slot.key is not None
            or slot.selection_leases
            or slot.async_pins
            or slot.pending_receipts
        ):
            raise RuntimeError("request state attempted to reuse an occupied slot")
        parameters = tuple(slot.module.parameters())
        buffers = tuple(slot.module.buffers())
        with torch.no_grad():
            for parameter, initial in zip(
                parameters, self._initial_parameters, strict=True
            ):
                parameter.copy_(initial)
                parameter.grad = None
            for buffer, initial in zip(
                buffers, self._initial_buffers, strict=True
            ):
                buffer.copy_(initial)
        slot.optimizer = optimizer
        slot.generation += 1
        slot.version = 0
        slot.key = key
        slot.aliases.clear()
        return StateSlotRef(slot_id, slot.generation)

    def _validate_selection(
        self, selection: RequestStateSelectionBatch
    ) -> _ActiveSelection:
        if selection.resource != self._resource or selection.epoch != self._epoch:
            raise RuntimeError("request state selection belongs to another resource epoch")
        try:
            active = self._active_selections[selection.selection_id]
        except KeyError as exc:
            raise RuntimeError("request state selection is no longer active") from exc
        if selection.rows != active.rows or selection.leases != active.leases:
            raise RuntimeError("request state selection leases differ from admission")
        if (
            selection.device_slot_ids.dtype is not torch.long
            or selection.device_slot_ids.device != self._device
            or tuple(selection.device_slot_ids.shape) != (len(active.slot_ids),)
            or selection.valid_mask.dtype is not torch.bool
            or selection.valid_mask.device != self._device
            or tuple(selection.valid_mask.shape) != (len(active.slot_ids),)
            or not torch.equal(
                selection.device_slot_ids,
                torch.tensor(
                    active.slot_ids,
                    dtype=torch.long,
                    device=self._device,
                ),
            )
            or not bool(torch.all(selection.valid_mask).item())
        ):
            raise RuntimeError("request state selection tensors differ from admission")
        for lease in selection.leases:
            slot = self._slots[lease.slot.slot_id]
            if (
                slot.generation != lease.slot.generation
                or slot.key != lease.key
                or lease.epoch != self._epoch
            ):
                raise RuntimeError("request state selection contains a stale slot lease")
        return active

    def _require_audited_selection(
        self, selection: RequestStateSelectionBatch
    ) -> _ActiveSelection:
        active = self._validate_selection(selection)
        if any(
            self._slots[lease.slot.slot_id].pending_receipts
            for lease in active.leases
        ):
            raise RuntimeError("request state selection has an unaudited effect")
        return active

    def acquire(
        self, identities: tuple[RequestIdentity, ...]
    ) -> RequestStateAcquisition:
        if not identities:
            raise ValueError("request state selection requires identities")
        keys = tuple(
            RequestStateKey.project(identity, self._key_fields)
            for identity in identities
        )
        unique_keys = tuple(dict.fromkeys(keys))
        with self._lock:
            self._require_live()
            for identity, key in zip(identities, keys, strict=True):
                prior = self._key_by_identity.get(identity)
                if prior is not None and prior != key:
                    raise RuntimeError("request identity changed its signed state key")
            new_keys = tuple(
                key for key in unique_keys if key not in self._slot_by_key
            )
            free_slots = tuple(
                slot_id
                for slot_id, slot in enumerate(self._slots)
                if (
                    slot.key is None
                    and not slot.selection_leases
                    and not slot.async_pins
                    and not slot.pending_receipts
                )
            )
            if len(new_keys) > len(free_slots):
                raise RuntimeError(
                    "request trainable state capacity exhausted: "
                    f"required={len(new_keys)}, available={len(free_slots)}"
                )

            planned_slots = dict(self._slot_by_key)
            planned_optimizers: dict[int, torch.optim.Optimizer] = {}
            for key, slot_id in zip(new_keys, free_slots, strict=False):
                self._validate_slot_layout(slot_id)
                planned_slots[key] = slot_id
                planned_optimizers[slot_id] = self._optimizer_factory(
                    self._slots[slot_id].module
                )
            planned_slot_ids = tuple(planned_slots[key] for key in keys)
            slot_ids = torch.tensor(
                planned_slot_ids,
                dtype=torch.long,
                device=self._device,
            )
            valid_mask = torch.ones(
                (len(keys),), dtype=torch.bool, device=self._device
            )

            admitted: list[StateSlotEffect] = []
            try:
                for key, slot_id in zip(new_keys, free_slots, strict=False):
                    slot_ref = self._reset_slot(
                        slot_id, key, planned_optimizers[slot_id]
                    )
                    self._slot_by_key[key] = slot_id
                    admitted.append(StateSlotEffect(key, slot_ref, 0, 0))

                for identity, key in zip(identities, keys, strict=True):
                    self._key_by_identity[identity] = key
                    slot = self._slots[self._slot_by_key[key]]
                    slot.aliases.add(identity)

                leases = tuple(
                    RequestStateLease(
                        key=key,
                        slot=StateSlotRef(
                            self._slot_by_key[key],
                            self._slots[self._slot_by_key[key]].generation,
                        ),
                        epoch=self._epoch,
                    )
                    for key in unique_keys
                )
                lease_by_key = {lease.key: lease for lease in leases}
                rows = tuple(
                    RequestStateSelectionRow(identity, lease_by_key[key])
                    for identity, key in zip(identities, keys, strict=True)
                )
                selection_id = self._next_selection_id
                self._next_selection_id += 1
                active = _ActiveSelection(
                    rows=rows,
                    leases=leases,
                    slot_ids=tuple(row.lease.slot.slot_id for row in rows),
                )
                self._active_selections[selection_id] = active
                for lease in active.leases:
                    self._slots[lease.slot.slot_id].selection_leases += 1
            except Exception:
                self._poisoned = True
                raise

            selection = RequestStateSelectionBatch(
                resource=self._resource,
                selection_id=selection_id,
                epoch=self._epoch,
                rows=rows,
                leases=leases,
                device_slot_ids=slot_ids,
                valid_mask=valid_mask,
            )
            return RequestStateAcquisition(
                selection=selection,
                admission=self._issue_receipt(
                    StateEffectOperation.ADMIT,
                    selection_id,
                    tuple(admitted),
                ),
            )

    def groups(
        self, selection: RequestStateSelectionBatch
    ) -> tuple[RequestStateModuleGroup[ModuleT], ...]:
        with self._lock:
            self._require_live()
            self._require_audited_selection(selection)
            groups: list[RequestStateModuleGroup[ModuleT]] = []
            for lease in selection.leases:
                positions = [
                    position
                    for position, row in enumerate(selection.rows)
                    if row.lease == lease
                ]
                groups.append(
                    RequestStateModuleGroup(
                        lease=lease,
                        row_indices=torch.tensor(
                            positions,
                            dtype=torch.long,
                            device=selection.device_slot_ids.device,
                        ),
                        module=self._slots[lease.slot.slot_id].module,
                    )
                )
            return tuple(groups)

    def read_receipt(
        self, selection: RequestStateSelectionBatch
    ) -> StateEffectReceipt:
        with self._lock:
            self._require_live()
            self._require_audited_selection(selection)
            effects = tuple(
                StateSlotEffect(
                    lease.key,
                    lease.slot,
                    self._slots[lease.slot.slot_id].version,
                    self._slots[lease.slot.slot_id].version,
                )
                for lease in selection.leases
            )
            return self._issue_receipt(
                StateEffectOperation.READ,
                selection.selection_id,
                effects,
            )

    def optimize(
        self,
        selection: RequestStateSelectionBatch,
        losses: tuple[torch.Tensor, ...],
    ) -> StateEffectReceipt:
        with self._lock:
            self._require_live()
            self._require_audited_selection(selection)
            if len(losses) != len(selection.leases):
                raise ValueError("request state losses must align with unique leases")
            if any(loss.numel() != 1 for loss in losses):
                raise ValueError("request state optimizer loss must be scalar")
            effects: list[StateSlotEffect] = []
            try:
                for lease, loss in zip(selection.leases, losses, strict=True):
                    slot = self._slots[lease.slot.slot_id]
                    before = slot.version
                    with torch.inference_mode(False), torch.enable_grad():
                        slot.optimizer.zero_grad(set_to_none=True)
                        torch.autograd.backward(loss)
                        slot.optimizer.step()
                    slot.version += 1
                    effects.append(
                        StateSlotEffect(
                            lease.key,
                            lease.slot,
                            before,
                            slot.version,
                        )
                    )
            except Exception:
                self._poisoned = True
                raise
            return self._issue_receipt(
                StateEffectOperation.TRANSITION,
                selection.selection_id,
                tuple(effects),
            )

    def release_selection(self, selection: RequestStateSelectionBatch) -> None:
        with self._lock:
            active = self._require_audited_selection(selection)
            for lease in active.leases:
                slot = self._slots[lease.slot.slot_id]
                slot.selection_leases -= 1
            self._active_selections.pop(selection.selection_id)

    def pin_synchronous(
        self, selection: RequestStateSelectionBatch
    ) -> RequestStateAsyncPin:
        with self._lock:
            self._require_live()
            if self._device.type == "cuda":
                raise RuntimeError(
                    "CUDA request state pins require a recorded completion event"
                )
            active = self._require_audited_selection(selection)
            pin_id = self._next_pin_id
            self._next_pin_id += 1
            slots = tuple(lease.slot for lease in active.leases)
            self._active_pins[pin_id] = _ActivePin(slots, None)
            for slot_ref in slots:
                self._slots[slot_ref.slot_id].async_pins += 1
            return RequestStateAsyncPin(
                resource=self._resource,
                pin_id=pin_id,
                epoch=self._epoch,
                slots=slots,
            )

    def pin_cuda(
        self,
        selection: RequestStateSelectionBatch,
        stream: torch.cuda.Stream,
    ) -> RequestStateAsyncPin:
        """Record the exact stream tail that guards selected slot reuse."""

        with self._lock:
            self._require_live()
            if self._device.type != "cuda":
                raise RuntimeError("CPU request state cannot record a CUDA pin")
            if stream.device != self._device:
                raise RuntimeError("request state completion stream uses another device")
            active = self._validate_selection(selection)
            event = torch.cuda.Event()
            event.record(stream)
            pin_id = self._next_pin_id
            self._next_pin_id += 1
            slots = tuple(lease.slot for lease in active.leases)
            self._active_pins[pin_id] = _ActivePin(slots, event)
            for slot_ref in slots:
                self._slots[slot_ref.slot_id].async_pins += 1
            return RequestStateAsyncPin(
                resource=self._resource,
                pin_id=pin_id,
                epoch=self._epoch,
                slots=slots,
            )

    def release_async_pin(self, pin: RequestStateAsyncPin) -> None:
        with self._lock:
            if pin.resource != self._resource or pin.epoch != self._epoch:
                raise RuntimeError("request state async pin belongs to another epoch")
            try:
                active = self._active_pins[pin.pin_id]
            except KeyError as exc:
                raise RuntimeError("request state async pin is no longer active") from exc
            if pin.slots != active.slots:
                raise RuntimeError("request state async pin slots differ")
            if active.completion is not None and not active.completion.query():
                raise RuntimeError("request state CUDA completion is still pending")
            for slot_ref in active.slots:
                slot = self._slots[slot_ref.slot_id]
                if slot.generation != slot_ref.generation:
                    raise RuntimeError("request state async pin contains a stale slot")
                slot.async_pins -= 1
            self._active_pins.pop(pin.pin_id)

    def wait_async_pin(self, pin: RequestStateAsyncPin) -> None:
        """Wait for a recorded completion fence, then release its slot pin."""

        with self._lock:
            try:
                active = self._active_pins[pin.pin_id]
            except KeyError as exc:
                raise RuntimeError("request state async pin is no longer active") from exc
            completion = active.completion
        if completion is not None:
            completion.synchronize()
        self.release_async_pin(pin)

    def retire(self, batch: RequestFinishedBatch) -> StateEffectReceipt:
        if len(batch.requests) != len(set(batch.requests)):
            raise ValueError("request-finished batch contains duplicate identities")
        with self._lock:
            self._require_live()
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
                slot = self._slots[self._slot_by_key[key]]
                if not slot.aliases.difference(retiring):
                    if (
                        slot.selection_leases
                        or slot.async_pins
                        or slot.pending_receipts
                    ):
                        raise RuntimeError(
                            "cannot retire request state with active leases, pins, or receipts"
                        )
                    released_keys.append(key)

            for identity in matched:
                key = self._key_by_identity.pop(identity)
                slot = self._slots[self._slot_by_key[key]]
                slot.aliases.remove(identity)

            effects: list[StateSlotEffect] = []
            for key in released_keys:
                slot_id = self._slot_by_key.pop(key)
                slot = self._slots[slot_id]
                slot_ref = StateSlotRef(slot_id, slot.generation)
                effects.append(
                    StateSlotEffect(
                        key,
                        slot_ref,
                        slot.version,
                        slot.version,
                    )
                )
                slot.key = None
                slot.aliases.clear()
            return self._issue_receipt(
                StateEffectOperation.RETIRE,
                0,
                tuple(effects),
            )

    def versions(self) -> tuple[RequestStateVersion, ...]:
        with self._lock:
            return tuple(
                RequestStateVersion(
                    key=key,
                    slot=StateSlotRef(slot_id, self._slots[slot_id].generation),
                    version=self._slots[slot_id].version,
                )
                for key, slot_id in sorted(
                    self._slot_by_key.items(), key=lambda item: item[1]
                )
            )

    def replace_epoch(self, epoch: int) -> None:
        if epoch <= self._epoch:
            raise ValueError("request state epoch replacement must advance")
        with self._lock:
            self._require_live()
            if (
                self._slot_by_key
                or self._active_selections
                or self._active_pins
                or self._issued_receipts
            ):
                raise RuntimeError("cannot replace request state epoch while requests are active")
            self._epoch = epoch


__all__ = [
    "RequestStateAcquisition",
    "RequestStateAsyncPin",
    "RequestStateKey",
    "RequestStateLease",
    "RequestStateModuleGroup",
    "RequestStateSelectionBatch",
    "RequestStateSelectionRow",
    "RequestStateVersion",
    "RequestTrainableStateStore",
    "StateEffectOperation",
    "StateEffectReceipt",
    "StateSlotEffect",
    "StateSlotRef",
]
