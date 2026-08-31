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


@dataclass(frozen=True, slots=True, order=True)
class StateSlotRef:
    slot_id: int
    generation: int

    def __post_init__(self) -> None:
        if self.slot_id < 0:
            raise ValueError("request state slot id must be non-negative")
        if self.generation < 1:
            raise ValueError("request state slot generation must be positive")

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
    before_versions: tuple[int, ...]
    device_slot_ids: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class RequestStateLossBatch:
    """One closed optimizer payload aligned to a request-state selection."""

    selection: RequestStateSelectionBatch
    losses: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.losses) != len(self.selection.leases):
            raise ValueError(
                "request-state loss batch must align with unique selection leases"
            )
        if any(loss.numel() != 1 for loss in self.losses):
            raise ValueError("request-state loss batch entries must be scalar")


@dataclass(frozen=True, slots=True)
class RequestIdentityBatch:
    """One exact backend-resolved identity batch for a native step.

    The backend boundary stages this value once.  Request-state resources use
    the same rows but independently project their signed key declaration.
    """

    identities: tuple[RequestIdentity, ...]

    def __post_init__(self) -> None:
        if any(
            not identity.request_id
            or identity.prompt_idx < 0
            or identity.sample_idx < 0
            for identity in self.identities
        ):
            raise ValueError(
                "request identity batch requires fully resolved identities"
            )


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
class RequestStateEffectAudit:
    """Typed before-image for one signed request-state Region effect."""

    resource: str
    epoch: int
    step_id: int
    selection: RequestStateSelectionBatch
    operation: StateEffectOperation
    selected_before: tuple[StateSlotEffect, ...]
    store_before: tuple[RequestStateVersion, ...]


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
    canonical_row_indices: torch.Tensor
    before_version: int
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
    before_versions: tuple[int, ...]
    slot_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ActivePin:
    slots: tuple[StateSlotRef, ...]
    completion: torch.cuda.Event | None


class RequestTrainableStateStore(Generic[ModuleT]):
    """A fixed slot bank with generation-safe request ownership.

    Modules are allocated during cold specialization and keep their parameter
    addresses for the store lifetime.  Physical allocation does not advance
    the logical initializer RNG; each newly admitted key invokes the declared
    module factory and copies that fresh state into its fixed slot. Optimizers
    are intentionally eager in this first capability: slot reuse reconstructs
    optimizer state, so Rust must not classify request-scoped transitions as
    capture-compatible yet.
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

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None
        )
        try:
            modules = tuple(module_factory() for _slot_id in range(capacity))
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)
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
        storage_addresses: set[int] = set()
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
            module_storage_addresses = {
                tensor.untyped_storage().data_ptr()
                for tensor in (*parameters, *buffers)
                if tensor.numel()
            }
            if (
                parameter_ids.intersection(module_parameter_ids)
                or parameter_addresses.intersection(module_parameter_addresses)
                or buffer_ids.intersection(module_buffer_ids)
                or buffer_addresses.intersection(module_buffer_addresses)
                or storage_addresses.intersection(module_storage_addresses)
            ):
                raise RuntimeError(
                    "request state module slots must not share tensor storage"
                )
            parameter_ids.update(module_parameter_ids)
            parameter_addresses.update(module_parameter_addresses)
            buffer_ids.update(module_buffer_ids)
            buffer_addresses.update(module_buffer_addresses)
            storage_addresses.update(module_storage_addresses)
        self._resource = resource
        self._key_fields = key_fields
        self._capacity = capacity
        self._epoch = epoch
        self._device = device
        self._module_factory = module_factory
        self._optimizer_factory = optimizer_factory
        self._initial_parameters = canonical_parameters
        self._initial_buffers = canonical_buffers
        self._physical_parameter_ids = parameter_ids
        self._physical_parameter_addresses = parameter_addresses
        self._physical_buffer_ids = buffer_ids
        self._physical_buffer_addresses = buffer_addresses
        self._physical_storage_addresses = storage_addresses
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else None
        )
        try:
            self._slots = [
                _RequestStateSlot(module, optimizer_factory(module))
                for module in modules
            ]
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, device)
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

    @property
    def device(self) -> torch.device:
        """The fixed device whose stream owns this store's slot tensors."""

        return self._device

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
        initial_parameters: tuple[torch.nn.Parameter, ...],
        initial_buffers: tuple[torch.Tensor, ...],
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
                parameters, initial_parameters, strict=True
            ):
                parameter.copy_(initial)
                parameter.grad = None
            for buffer, initial in zip(
                buffers, initial_buffers, strict=True
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
        if (
            selection.rows != active.rows
            or selection.leases != active.leases
            or selection.before_versions != active.before_versions
        ):
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
            planned_initializations: dict[
                int,
                tuple[
                    tuple[torch.nn.Parameter, ...],
                    tuple[torch.Tensor, ...],
                ],
            ] = {}
            for key, slot_id in zip(new_keys, free_slots, strict=False):
                self._validate_slot_layout(slot_id)
                with torch.inference_mode(False), torch.enable_grad():
                    initial_module = self._module_factory()
                    initial_parameters = tuple(initial_module.parameters())
                    initial_buffers = tuple(initial_module.buffers())
                if len(initial_parameters) != len(self._initial_parameters) or any(
                    tuple(parameter.shape) != tuple(canonical.shape)
                    or parameter.dtype is not canonical.dtype
                    or parameter.device != canonical.device
                    for parameter, canonical in zip(
                        initial_parameters,
                        self._initial_parameters,
                        strict=True,
                    )
                ):
                    raise RuntimeError(
                        "request state admission parameter layouts differ"
                    )
                if len(initial_buffers) != len(self._initial_buffers) or any(
                    tuple(buffer.shape) != tuple(canonical.shape)
                    or buffer.dtype is not canonical.dtype
                    or buffer.device != canonical.device
                    for buffer, canonical in zip(
                        initial_buffers,
                        self._initial_buffers,
                        strict=True,
                    )
                ):
                    raise RuntimeError(
                        "request state admission buffer layouts differ"
                    )
                if (
                    any(
                        id(parameter) in self._physical_parameter_ids
                        or parameter.data_ptr()
                        in self._physical_parameter_addresses
                        for parameter in initial_parameters
                    )
                    or any(
                        id(buffer) in self._physical_buffer_ids
                        or (
                            buffer.numel() > 0
                            and buffer.data_ptr()
                            in self._physical_buffer_addresses
                        )
                        for buffer in initial_buffers
                    )
                    or any(
                        tensor.numel() > 0
                        and tensor.untyped_storage().data_ptr()
                        in self._physical_storage_addresses
                        for tensor in (*initial_parameters, *initial_buffers)
                    )
                ):
                    raise RuntimeError(
                        "request state admission initializer aliases slot storage"
                    )
                planned_slots[key] = slot_id
                planned_initializations[slot_id] = (
                    initial_parameters,
                    initial_buffers,
                )
                with torch.inference_mode(False), torch.enable_grad():
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
                    initial_parameters, initial_buffers = (
                        planned_initializations[slot_id]
                    )
                    slot_ref = self._reset_slot(
                        slot_id,
                        key,
                        planned_optimizers[slot_id],
                        initial_parameters,
                        initial_buffers,
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
                    before_versions=tuple(
                        self._slots[lease.slot.slot_id].version
                        for lease in leases
                    ),
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
                before_versions=active.before_versions,
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
            for lease, before_version in zip(
                selection.leases,
                selection.before_versions,
                strict=True,
            ):
                positions = [
                    position
                    for position, row in enumerate(selection.rows)
                    if row.lease == lease
                ]
                groups.append(
                    RequestStateModuleGroup(
                        lease=lease,
                        canonical_row_indices=torch.tensor(
                            positions,
                            dtype=torch.long,
                            device=selection.device_slot_ids.device,
                        ),
                        before_version=before_version,
                        module=self._slots[lease.slot.slot_id].module,
                    )
                )
            return tuple(groups)

    def selection_versions(
        self, selection: RequestStateSelectionBatch
    ) -> tuple[RequestStateVersion, ...]:
        """Return the exact live versions selected by one active lease batch.

        This is intentionally a typed, non-mutating observation.  Runtime
        audits use it to form their before image; it never manufactures a
        receipt or allows the legacy Hashable snapshot registry to inspect a
        request resource.
        """

        with self._lock:
            self._require_live()
            active = self._require_audited_selection(selection)
            return tuple(
                RequestStateVersion(
                    key=lease.key,
                    slot=lease.slot,
                    version=self._slots[lease.slot.slot_id].version,
                )
                for lease in active.leases
            )

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

    def async_pin_complete(self, pin: RequestStateAsyncPin) -> bool:
        """Query one exact completion token without releasing its slot pin."""

        with self._lock:
            if pin.resource != self._resource or pin.epoch != self._epoch:
                raise RuntimeError(
                    "request state async pin belongs to another epoch"
                )
            try:
                active = self._active_pins[pin.pin_id]
            except KeyError as exc:
                raise RuntimeError(
                    "request state async pin is no longer active"
                ) from exc
            if pin.slots != active.slots:
                raise RuntimeError("request state async pin slots differ")
            completion = active.completion
            return completion is None or completion.query()

    def order_async_pin(
        self,
        pin: RequestStateAsyncPin,
        stream: torch.cuda.Stream,
    ) -> None:
        """Order one Rust-selected stream after a prior state completion."""

        with self._lock:
            if pin.resource != self._resource or pin.epoch != self._epoch:
                raise RuntimeError(
                    "request state async pin belongs to another epoch"
                )
            try:
                active = self._active_pins[pin.pin_id]
            except KeyError as exc:
                raise RuntimeError(
                    "request state async pin is no longer active"
                ) from exc
            if pin.slots != active.slots:
                raise RuntimeError("request state async pin slots differ")
            if stream.device != self._device:
                raise RuntimeError(
                    "request state ordering stream uses another device"
                )
            completion = active.completion
            if completion is None:
                raise RuntimeError("CUDA request state pin has no completion event")
            stream.wait_event(completion)
            for slot_ref in active.slots:
                slot = self._slots[slot_ref.slot_id]
                if slot.generation != slot_ref.generation:
                    raise RuntimeError(
                        "request state async pin contains a stale slot"
                    )
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


@dataclass(slots=True)
class _StagedRequestStateStep:
    identities: RequestIdentityBatch
    selection: RequestStateSelectionBatch | None = None


class _RequestStateRuntimeCore(Generic[ModuleT]):
    """Shared mutable authority behind locally and composition-qualified views."""

    __slots__ = (
        "store",
        "steps",
        "transition_receipts",
        "completion_pins",
        "lock",
    )

    def __init__(self, store: RequestTrainableStateStore[ModuleT]) -> None:
        self.store = store
        self.steps: dict[int, _StagedRequestStateStep] = {}
        self.transition_receipts: dict[int, StateEffectReceipt] = {}
        self.completion_pins: dict[int, list[RequestStateAsyncPin]] = {}
        self.lock = RLock()


class RequestStateRuntimeService(Generic[ModuleT]):
    """The sole runtime authority for one request-state resource.

    This service is deliberately a concrete typed object, rather than a
    provider callback.  It owns identity staging, one acquisition per step,
    exact receipt consumption, and selection retirement.  The initial CUDA
    policy is fail-closed: native execution has not yet proved the stream
    completion fence required before reusing a request slot.
    """

    __slots__ = ("_core", "_resource")

    def __init__(
        self,
        store: RequestTrainableStateStore[ModuleT],
        *,
        signed_resource: str | None = None,
        _core: _RequestStateRuntimeCore[ModuleT] | None = None,
    ) -> None:
        if signed_resource is not None and not signed_resource:
            raise ValueError("signed request-state resource must be non-empty")
        if _core is not None and _core.store is not store:
            raise RuntimeError("request-state service core belongs to another store")
        self._core = (
            _RequestStateRuntimeCore(store) if _core is None else _core
        )
        self._resource = store.resource if signed_resource is None else signed_resource

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
    def _lock(self):
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
        """Read the service-owned version view without exposing the store."""

        with self._lock:
            return self._store.versions()

    def for_signed_resource(
        self, resource: str
    ) -> "RequestStateRuntimeService[ModuleT]":
        """Bind this cold store to one explicitly qualified schedule resource."""

        if not resource:
            raise ValueError("signed request-state resource must be non-empty")
        with self._lock:
            if self._steps or self._transition_receipts or self._completion_pins:
                raise RuntimeError(
                    "cannot qualify request-state service while it is active"
                )
            if resource == self.resource:
                return self
            return RequestStateRuntimeService(
                self._store,
                signed_resource=resource,
                _core=self._core,
            )

    def stage_identities(self, step_id: int, batch: RequestIdentityBatch) -> None:
        """Stage one immutable identity batch before any Region resolves it."""

        if step_id < 0:
            raise ValueError("request-state step id must be non-negative")
        with self._lock:
            self._collect_completed_pins()
            existing = self._steps.get(step_id)
            if existing is None:
                self._steps[step_id] = _StagedRequestStateStep(batch)
                return
            if existing.identities != batch:
                raise RuntimeError(
                    "request-state step identity batch differs from its staged batch"
                )

    def _acquire_selection(self, step_id: int) -> RequestStateSelectionBatch:
        """Acquire the exact resource projection once for a staged step."""

        with self._lock:
            try:
                staged = self._steps[step_id]
            except KeyError as exc:
                raise RuntimeError(
                    "request-state identities must be staged before selection acquisition"
                ) from exc
            if staged.selection is not None:
                return staged.selection
            acquired = self._store.acquire(staged.identities.identities)
            self._validate_admission(acquired)
            self._store.consume_receipt(acquired.admission)
            staged.selection = acquired.selection
            return acquired.selection

    def selection_for_step(
        self,
        step_id: int,
        stream: torch.cuda.Stream | None = None,
    ) -> RequestStateSelectionBatch:
        """Resolve one non-empty selection on its signed execution stream."""

        selection = self.selection_value_for_step(step_id, stream)
        if selection is None:
            raise RuntimeError("request-state identity batch is empty")
        return selection

    def selection_value_for_step(
        self,
        step_id: int,
        stream: torch.cuda.Stream | None = None,
    ) -> RequestStateSelectionBatch | None:
        """Resolve an explicit empty identity batch to a skipped Region input."""

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
        """Capture a typed before image for a signed read or transition."""

        with self._lock:
            self._require_staged_selection(step_id, selection)
            self._validate_operation(operation, selection)
            selected = tuple(
                StateSlotEffect(
                    version.key,
                    version.slot,
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
        """Run one service-owned strict transition for the selected slots."""

        with self._lock:
            self._require_selection(selection)
            if selection.selection_id in self._transition_receipts:
                raise RuntimeError(
                    "request-state selection already has an unaudited transition"
                )
            receipt = self._store.optimize(selection, losses)
            self._transition_receipts[selection.selection_id] = receipt
            return receipt

    def groups(
        self, selection: RequestStateSelectionBatch
    ) -> tuple[RequestStateModuleGroup[ModuleT], ...]:
        """Expose selected modules without exposing a second execution authority."""

        with self._lock:
            self._require_selection(selection)
            return self._store.groups(selection)

    def after(self, audit: RequestStateEffectAudit) -> None:
        """Validate and consume the exact receipt for a previously begun effect."""

        with self._lock:
            self._require_staged_selection(audit.step_id, audit.selection)
            if audit.resource != self.resource or audit.epoch != self.epoch:
                raise RuntimeError("request-state audit belongs to another resource epoch")
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
                    "request-state runtime supports only strict read and transition effects"
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
        """Pin selected slots to the exact CUDA stream tail of one effect."""

        with self._lock:
            self._require_staged_selection(step_id, selection)
            if self._store.device.type != "cuda":
                raise RuntimeError(
                    "CPU request state cannot record a CUDA completion"
                )
            pin = self._store.pin_cuda(selection, stream)
            self._completion_pins.setdefault(step_id, []).append(pin)

    def finish_step(self, step_id: int) -> None:
        """Release this resource's exact step selection at executor retirement."""

        with self._lock:
            try:
                staged = self._steps[step_id]
            except KeyError:
                return
            selection = staged.selection
            if selection is None:
                self._steps.pop(step_id)
                return
            if selection.selection_id in self._transition_receipts:
                raise RuntimeError(
                    "request-state step retires with an unaudited transition receipt"
                )
            self._store.release_selection(selection)
            self._steps.pop(step_id)

    def retire_requests(
        self, batch: RequestFinishedBatch
    ) -> StateEffectReceipt:
        """Retire completed identities only after the service reaches a safe point."""

        with self._lock:
            if self._steps or self._transition_receipts:
                raise RuntimeError(
                    "request-state retirement requires no active step, selection, "
                    "or unaudited receipt"
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
                or len({effect.slot for effect in receipt.slots}) != len(receipt.slots)
            ):
                raise RuntimeError("request-state retirement receipt has the wrong identity")
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
        matching_steps = tuple(
            step_id
            for step_id, staged in self._steps.items()
            if staged.selection is selection
        )
        if len(matching_steps) != 1:
            raise RuntimeError("request-state selection is not owned by this runtime service")

    def _require_staged_selection(
        self,
        step_id: int,
        selection: RequestStateSelectionBatch,
    ) -> None:
        try:
            staged = self._steps[step_id]
        except KeyError as exc:
            raise RuntimeError("request-state step is not staged") from exc
        if staged.selection is not selection:
            raise RuntimeError(
                "request-state audit selection differs from the staged step selection"
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
            raise RuntimeError("request-state admission receipt has the wrong identity")
        leases = tuple(row.lease for row in selection.rows)
        lease_by_slot = {lease.slot: lease for lease in selection.leases}
        if any(lease_by_slot.get(lease.slot) != lease for lease in leases):
            raise RuntimeError("request-state admission selection has an invalid slot lease")
        for effect in receipt.slots:
            expected = lease_by_slot.get(effect.slot)
            if (
                expected is None
                or effect.key != expected.key
                or effect.before_version != 0
                or effect.after_version != 0
            ):
                raise RuntimeError("request-state admission receipt differs from selection")

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
                "request-state runtime supports only strict read and transition effects"
            )
        if (
            selection.resource != self._store.resource
            or selection.epoch != self.epoch
        ):
            raise RuntimeError("request-state selection belongs to another resource epoch")

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
            raise RuntimeError("request-state effect receipt has the wrong identity")
        if tuple(effect.slot for effect in receipt.slots) != tuple(
            effect.slot for effect in audit.selected_before
        ):
            raise RuntimeError("request-state effect receipt slot generation differs")
        current = {version.slot: version for version in self._store.versions()}
        before_all = {version.slot: version for version in audit.store_before}
        if set(current) != set(before_all):
            raise RuntimeError(
                "request-state slot identity changed while a Region effect executed"
            )
        selected_slots = {effect.slot for effect in audit.selected_before}
        if len(selected_slots) != len(audit.selected_before):
            raise RuntimeError("request-state selection contains duplicate slot leases")
        expected_delta = 0 if operation is StateEffectOperation.READ else 1
        for before, effect in zip(
            audit.selected_before, receipt.slots, strict=True
        ):
            expected_after = before.before_version + expected_delta
            current_version = current.get(before.slot)
            if (
                effect.key != before.key
                or effect.slot != before.slot
                or effect.before_version != before.before_version
                or effect.after_version != expected_after
                or current_version is None
                or current_version.key != before.key
                or current_version.version != expected_after
            ):
                raise RuntimeError(
                    "request-state selected slot violated its signed before/after effect"
                )
        for slot, before in before_all.items():
            if slot in selected_slots:
                continue
            current_version = current[slot]
            if current_version.key != before.key or current_version.version != before.version:
                raise RuntimeError(
                    "request-state unselected slot changed during a signed effect"
                )


__all__ = [
    "RequestStateAcquisition",
    "RequestStateAsyncPin",
    "RequestStateEffectAudit",
    "RequestIdentityBatch",
    "RequestStateKey",
    "RequestStateLease",
    "RequestStateLossBatch",
    "RequestStateModuleGroup",
    "RequestStateSelectionBatch",
    "RequestStateSelectionRow",
    "RequestStateVersion",
    "RequestStateRuntimeService",
    "RequestTrainableStateStore",
    "StateEffectOperation",
    "StateEffectReceipt",
    "StateSlotEffect",
    "StateSlotRef",
]
