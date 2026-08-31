"""Resident physical storage for request-slot trainable modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from tllm.contracts.request_state import (
    RequestStateAsyncPin,
    RequestStateLease,
    RequestStateSelectionBatch,
)
from tllm.runtime.request_slot_arena import (
    RequestSlotAdmissionPlan,
    RequestSlotPinPlan,
    RequestSlotTransitionPlan,
)


ModuleT = TypeVar("ModuleT", bound=torch.nn.Module)


@dataclass(frozen=True, slots=True)
class RequestStateModuleGroup(Generic[ModuleT]):
    lease: RequestStateLease
    canonical_row_indices: torch.Tensor
    before_version: int
    module: ModuleT


@dataclass(frozen=True, slots=True)
class NoCompletionFence:
    pass


@dataclass(frozen=True, slots=True)
class CudaCompletionFence:
    event: torch.cuda.Event


CompletionFence = NoCompletionFence | CudaCompletionFence


@dataclass(slots=True)
class _ResidentSlot(Generic[ModuleT]):
    module: ModuleT
    optimizer: torch.optim.Optimizer


@dataclass(frozen=True, slots=True)
class _PreparedResidentSlot:
    slot_id: int
    parameters: tuple[torch.nn.Parameter, ...]
    buffers: tuple[torch.Tensor, ...]
    optimizer: torch.optim.Optimizer


@dataclass(frozen=True, slots=True)
class ResidentAdmissionPreparation:
    transaction_id: int
    slots: tuple[_PreparedResidentSlot, ...]


@dataclass(frozen=True, slots=True)
class _CpuRngState:
    cpu: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CudaRngState:
    cpu: torch.Tensor
    cuda: torch.Tensor
    device: torch.device


_TorchRngState = _CpuRngState | _CudaRngState


class ResidentModuleAdapter(Generic[ModuleT]):
    """Own modules, optimizers, buffers, and completion events.

    Logical identity arrives only through an arena-issued plan and is never
    persisted here. A failed physical commit permanently poisons the adapter
    so partially mutated storage cannot be observed as live state.
    """

    __slots__ = (
        "_capacity",
        "_device",
        "_module_factory",
        "_optimizer_factory",
        "_slots",
        "_layout_parameters",
        "_layout_buffers",
        "_physical_parameter_ids",
        "_physical_parameter_addresses",
        "_physical_buffer_ids",
        "_physical_buffer_addresses",
        "_physical_storage_addresses",
        "_fences",
        "_poisoned",
    )

    def __init__(
        self,
        *,
        capacity: int,
        device: torch.device,
        module_factory: Callable[[], ModuleT],
        optimizer_factory: Callable[[ModuleT], torch.optim.Optimizer],
    ) -> None:
        if capacity < 1:
            raise ValueError("resident module adapter capacity must be positive")
        self._capacity = capacity
        self._device = device
        self._module_factory = module_factory
        self._optimizer_factory = optimizer_factory

        rng_state = self._capture_rng_state()
        try:
            with torch.inference_mode(False), torch.enable_grad():
                modules = tuple(module_factory() for _slot_id in range(capacity))
                optimizers = tuple(
                    optimizer_factory(module) for module in modules
                )
        finally:
            self._restore_rng_state(rng_state)
        if len({id(module) for module in modules}) != capacity:
            raise RuntimeError("resident module slots must be distinct")

        layout_parameters = tuple(
            parameter.detach().clone() for parameter in modules[0].parameters()
        )
        layout_buffers = tuple(
            buffer.detach().clone() for buffer in modules[0].buffers()
        )
        if not layout_parameters:
            raise ValueError("resident modules require parameters")

        parameter_ids: set[int] = set()
        parameter_addresses: set[int] = set()
        buffer_ids: set[int] = set()
        buffer_addresses: set[int] = set()
        storage_addresses: set[int] = set()
        for module in modules:
            parameters = tuple(module.parameters())
            buffers = tuple(module.buffers())
            self._require_layout(
                parameters,
                buffers,
                layout_parameters,
                layout_buffers,
                label="resident module slot",
            )
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
                    "resident module slots must not share tensor storage"
                )
            parameter_ids.update(module_parameter_ids)
            parameter_addresses.update(module_parameter_addresses)
            buffer_ids.update(module_buffer_ids)
            buffer_addresses.update(module_buffer_addresses)
            storage_addresses.update(module_storage_addresses)

        self._slots = [
            _ResidentSlot(module, optimizer)
            for module, optimizer in zip(modules, optimizers, strict=True)
        ]
        self._layout_parameters = layout_parameters
        self._layout_buffers = layout_buffers
        self._physical_parameter_ids = parameter_ids
        self._physical_parameter_addresses = parameter_addresses
        self._physical_buffer_ids = buffer_ids
        self._physical_buffer_addresses = buffer_addresses
        self._physical_storage_addresses = storage_addresses
        self._fences: dict[int, CompletionFence] = {}
        self._poisoned = False

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def device(self) -> torch.device:
        return self._device

    def poison(self) -> None:
        self._poisoned = True

    def prepare_admission(
        self, plan: RequestSlotAdmissionPlan
    ) -> ResidentAdmissionPreparation:
        self._require_live()
        rng_state = self._capture_rng_state()
        prepared: list[_PreparedResidentSlot] = []
        try:
            for admission in plan.admissions:
                slot_id = admission.slot.slot_id
                self._validate_slot_layout(slot_id)
                with torch.inference_mode(False), torch.enable_grad():
                    initial_module = self._module_factory()
                    parameters = tuple(initial_module.parameters())
                    buffers = tuple(initial_module.buffers())
                    optimizer = self._optimizer_factory(
                        self._slots[slot_id].module
                    )
                self._require_layout(
                    parameters,
                    buffers,
                    self._layout_parameters,
                    self._layout_buffers,
                    label="resident admission initializer",
                )
                if self._aliases_resident_storage(parameters, buffers):
                    raise RuntimeError(
                        "resident admission initializer aliases slot storage"
                    )
                prepared.append(
                    _PreparedResidentSlot(
                        slot_id,
                        parameters,
                        buffers,
                        optimizer,
                    )
                )
        except Exception:
            self._restore_rng_state(rng_state)
            raise
        return ResidentAdmissionPreparation(
            plan.transaction_id,
            tuple(prepared),
        )

    def commit_admission(
        self,
        plan: RequestSlotAdmissionPlan,
        preparation: ResidentAdmissionPreparation,
    ) -> None:
        self._require_live()
        if (
            preparation.transaction_id != plan.transaction_id
            or tuple(slot.slot_id for slot in preparation.slots)
            != tuple(admission.slot.slot_id for admission in plan.admissions)
        ):
            raise RuntimeError(
                "resident admission preparation differs from its arena plan"
            )
        for prepared in preparation.slots:
            self._validate_slot_layout(prepared.slot_id)
            self._require_layout(
                prepared.parameters,
                prepared.buffers,
                self._layout_parameters,
                self._layout_buffers,
                label="resident admission preparation",
            )
        try:
            with torch.no_grad():
                for prepared in preparation.slots:
                    slot = self._slots[prepared.slot_id]
                    for parameter, initial in zip(
                        slot.module.parameters(),
                        prepared.parameters,
                        strict=True,
                    ):
                        parameter.copy_(initial)
                        parameter.grad = None
                    for buffer, initial in zip(
                        slot.module.buffers(),
                        prepared.buffers,
                        strict=True,
                    ):
                        buffer.copy_(initial)
                    slot.optimizer = prepared.optimizer
        except Exception:
            self._poisoned = True
            raise

    def groups(
        self, selection: RequestStateSelectionBatch
    ) -> tuple[RequestStateModuleGroup[ModuleT], ...]:
        self._require_live()
        groups: list[RequestStateModuleGroup[ModuleT]] = []
        for lease, before_version in zip(
            selection.leases,
            selection.before_versions,
            strict=True,
        ):
            positions = tuple(
                row.canonical_row_index
                for row in selection.rows
                if row.lease == lease
            )
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

    def execute_transition(
        self,
        plan: RequestSlotTransitionPlan,
        losses: tuple[torch.Tensor, ...],
    ) -> None:
        self._require_live()
        if len(losses) != len(plan.transitions):
            raise ValueError("request state losses must align with unique leases")
        if any(loss.numel() != 1 for loss in losses):
            raise ValueError("request state optimizer loss must be scalar")
        try:
            for effect, loss in zip(plan.transitions, losses, strict=True):
                slot = self._slots[effect.slot.slot_id]
                with torch.inference_mode(False), torch.enable_grad():
                    slot.optimizer.zero_grad(set_to_none=True)
                    torch.autograd.backward(loss)
                    slot.optimizer.step()
        except Exception:
            self._poisoned = True
            raise

    def record_synchronous_fence(self) -> NoCompletionFence:
        self._require_live()
        if self._device.type == "cuda":
            raise RuntimeError(
                "CUDA resident state pins require a recorded completion event"
            )
        return NoCompletionFence()

    def record_cuda_fence(
        self,
        stream: torch.cuda.Stream,
    ) -> CudaCompletionFence:
        self._require_live()
        if self._device.type != "cuda":
            raise RuntimeError("CPU resident state cannot record a CUDA pin")
        if stream.device != self._device:
            raise RuntimeError("resident completion stream uses another device")
        event = torch.cuda.Event()
        event.record(stream)
        return CudaCompletionFence(event)

    def commit_fence(
        self,
        plan: RequestSlotPinPlan,
        fence: CompletionFence,
    ) -> None:
        self._require_live()
        if plan.pin.pin_id in self._fences:
            raise RuntimeError("resident completion fence already exists")
        self._fences[plan.pin.pin_id] = fence

    def discard_fence(self, pin: RequestStateAsyncPin) -> None:
        self._require_live()
        try:
            self._fences.pop(pin.pin_id)
        except KeyError as exc:
            raise RuntimeError("resident completion fence is unknown") from exc

    def fence_complete(self, pin: RequestStateAsyncPin) -> bool:
        self._require_live()
        match self._fence(pin):
            case NoCompletionFence():
                return True
            case CudaCompletionFence(event=event):
                return event.query()

    def order_fence(
        self,
        pin: RequestStateAsyncPin,
        stream: torch.cuda.Stream,
    ) -> None:
        self._require_live()
        if stream.device != self._device:
            raise RuntimeError("resident ordering stream uses another device")
        match self._fence(pin):
            case NoCompletionFence():
                raise RuntimeError("CUDA resident pin has no completion event")
            case CudaCompletionFence(event=event):
                stream.wait_event(event)

    def wait_fence(self, pin: RequestStateAsyncPin) -> None:
        self._require_live()
        match self._fence(pin):
            case NoCompletionFence():
                return
            case CudaCompletionFence(event=event):
                event.synchronize()

    def _fence(self, pin: RequestStateAsyncPin) -> CompletionFence:
        try:
            return self._fences[pin.pin_id]
        except KeyError as exc:
            raise RuntimeError("resident completion fence is unknown") from exc

    def _validate_slot_layout(self, slot_id: int) -> None:
        slot = self._slots[slot_id]
        self._require_layout(
            tuple(slot.module.parameters()),
            tuple(slot.module.buffers()),
            self._layout_parameters,
            self._layout_buffers,
            label="resident module slot",
        )

    @staticmethod
    def _require_layout(
        parameters: tuple[torch.nn.Parameter, ...],
        buffers: tuple[torch.Tensor, ...],
        expected_parameters: tuple[torch.Tensor, ...],
        expected_buffers: tuple[torch.Tensor, ...],
        *,
        label: str,
    ) -> None:
        if len(parameters) != len(expected_parameters) or any(
            parameter.numel() < 1
            or tuple(parameter.shape) != tuple(expected.shape)
            or parameter.dtype is not expected.dtype
            or parameter.device != expected.device
            for parameter, expected in zip(
                parameters,
                expected_parameters,
                strict=True,
            )
        ):
            raise RuntimeError(f"{label} parameter layouts differ")
        if len(buffers) != len(expected_buffers) or any(
            tuple(buffer.shape) != tuple(expected.shape)
            or buffer.dtype is not expected.dtype
            or buffer.device != expected.device
            for buffer, expected in zip(
                buffers,
                expected_buffers,
                strict=True,
            )
        ):
            raise RuntimeError(f"{label} buffer layouts differ")

    def _aliases_resident_storage(
        self,
        parameters: tuple[torch.nn.Parameter, ...],
        buffers: tuple[torch.Tensor, ...],
    ) -> bool:
        return (
            any(
                id(parameter) in self._physical_parameter_ids
                or parameter.data_ptr() in self._physical_parameter_addresses
                for parameter in parameters
            )
            or any(
                id(buffer) in self._physical_buffer_ids
                or (
                    buffer.numel() > 0
                    and buffer.data_ptr() in self._physical_buffer_addresses
                )
                for buffer in buffers
            )
            or any(
                tensor.numel() > 0
                and tensor.untyped_storage().data_ptr()
                in self._physical_storage_addresses
                for tensor in (*parameters, *buffers)
            )
        )

    def _capture_rng_state(self) -> _TorchRngState:
        cpu = torch.get_rng_state()
        if self._device.type == "cuda":
            return _CudaRngState(
                cpu,
                torch.cuda.get_rng_state(self._device),
                self._device,
            )
        return _CpuRngState(cpu)

    @staticmethod
    def _restore_rng_state(state: _TorchRngState) -> None:
        match state:
            case _CpuRngState(cpu=cpu):
                torch.set_rng_state(cpu)
            case _CudaRngState(cpu=cpu, cuda=cuda, device=device):
                torch.set_rng_state(cpu)
                torch.cuda.set_rng_state(cuda, device)

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError("resident module adapter is poisoned")


__all__ = [
    "CompletionFence",
    "CudaCompletionFence",
    "NoCompletionFence",
    "RequestStateModuleGroup",
    "ResidentAdmissionPreparation",
    "ResidentModuleAdapter",
]
