"""Immutable-root physical storage for capacity-first request state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from tllm.contracts.request_state import (
    RequestStateSelectionBatch,
    RequestStateVersion,
    StateEffectOperation,
    StateEffectReceipt,
)
from tllm.runtime.paged_state import (
    AdmissionPageWrite,
    FrozenTensorField,
    FrozenTensorLayout,
    PageBundle,
    PagedStateVersionArena,
    StateLayoutId,
    StateResourceId,
    StateVersionLineage,
    StateVersionRef,
    StateVersionRootPin,
    TensorFieldId,
    TransitionPageWrite,
)
from tllm.runtime.request_slot_adapter import (
    CompletionFence,
    CudaCompletionFence,
    NoCompletionFence,
    RequestSlotAdapter,
)
from tllm.runtime.request_slot_arena import (
    RequestSlotAdmissionPlan,
    RequestSlotPinPlan,
    RequestSlotTransitionPlan,
)


@dataclass(frozen=True, slots=True)
class FixedBatchedSlotLayout:
    """Explicit physical-capacity-first tensors owned by one module."""

    capacity: int
    slot_parameters: tuple[torch.nn.Parameter, ...]
    slot_buffers: tuple[torch.Tensor, ...]


class FixedBatchedSlotModule(torch.nn.Module):
    """Nominal module whose parameters can be rebound to physical page pools."""

    def fixed_batched_slot_layout(self) -> FixedBatchedSlotLayout:
        raise NotImplementedError

    def install_fixed_batched_slot_layout(
        self,
        layout: FixedBatchedSlotLayout,
    ) -> None:
        raise NotImplementedError


BankT = TypeVar("BankT", bound=torch.nn.Module)


@dataclass(frozen=True, slots=True)
class FixedBatchedAdmissionPreparation:
    transaction_id: int
    slot_ids: tuple[int, ...]
    refs: tuple[StateVersionRef, ...]


@dataclass(frozen=True, slots=True)
class FixedBatchedSlotView(Generic[BankT]):
    selection: RequestStateSelectionBatch
    module: BankT
    row_slot_ids: torch.Tensor
    unique_slot_ids: torch.Tensor
    valid_mask: torch.Tensor
    before_versions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FixedBatchedAdamWState:
    step: torch.Tensor
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


class FixedBatchedAdamW:
    """AdamW that reads one immutable page and writes a distinct target page."""

    __slots__ = (
        "_parameters",
        "_lr",
        "_beta1",
        "_beta2",
        "_eps",
        "_weight_decay",
        "_states",
        "_gradients",
    )

    def __init__(
        self,
        layout: FixedBatchedSlotLayout,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0 or eps <= 0.0 or weight_decay < 0.0:
            raise ValueError(
                "fixed batched AdamW requires positive lr/eps and "
                "non-negative weight decay"
            )
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("fixed batched AdamW betas must be in [0, 1)")
        self._parameters = layout.slot_parameters
        self._lr = lr
        self._beta1 = beta1
        self._beta2 = beta2
        self._eps = eps
        self._weight_decay = weight_decay
        self._states = tuple(
            FixedBatchedAdamWState(
                step=torch.zeros(
                    (layout.capacity,),
                    dtype=torch.float32,
                    device=parameter.device,
                ),
                exp_avg=torch.zeros_like(parameter),
                exp_avg_sq=torch.zeros_like(parameter),
            )
            for parameter in layout.slot_parameters
        )
        self._gradients = tuple(
            torch.zeros_like(parameter) for parameter in layout.slot_parameters
        )
        for parameter, gradient in zip(
            self._parameters,
            self._gradients,
            strict=True,
        ):
            parameter.grad = gradient

    @property
    def states(self) -> tuple[FixedBatchedAdamWState, ...]:
        return self._states

    @property
    def state_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor
            for state in self._states
            for tensor in (state.step, state.exp_avg, state.exp_avg_sq)
        )

    def zero_grad(self) -> None:
        for gradient in self._gradients:
            gradient.zero_()

    def step(
        self,
        source_page_ids: torch.Tensor,
        target_page_ids: torch.Tensor,
    ) -> None:
        if (
            source_page_ids.ndim != 1
            or target_page_ids.ndim != 1
            or tuple(source_page_ids.shape) != tuple(target_page_ids.shape)
        ):
            raise ValueError("fixed batched AdamW page ids must be aligned")
        with torch.no_grad():
            for parameter, state, gradient in zip(
                self._parameters,
                self._states,
                self._gradients,
                strict=True,
            ):
                parameter_rows = parameter.index_select(0, source_page_ids)
                gradient_rows = gradient.index_select(0, source_page_ids)
                exp_avg_rows = state.exp_avg.index_select(0, source_page_ids)
                exp_avg_sq_rows = state.exp_avg_sq.index_select(
                    0,
                    source_page_ids,
                )
                step_rows = state.step.index_select(0, source_page_ids).add_(1)
                exp_avg_rows.mul_(self._beta1).add_(
                    gradient_rows,
                    alpha=1.0 - self._beta1,
                )
                exp_avg_sq_rows.mul_(self._beta2).addcmul_(
                    gradient_rows,
                    gradient_rows,
                    value=1.0 - self._beta2,
                )
                if self._weight_decay:
                    parameter_rows.mul_(1.0 - self._lr * self._weight_decay)
                bias_correction1 = 1.0 - torch.pow(
                    self._beta1,
                    step_rows,
                )
                bias_correction2 = 1.0 - torch.pow(
                    self._beta2,
                    step_rows,
                )
                row_shape = (int(source_page_ids.shape[0]),) + (1,) * (
                    parameter_rows.ndim - 1
                )
                step_size = self._lr / bias_correction1
                denominator = (
                    exp_avg_sq_rows.sqrt()
                    / bias_correction2.sqrt().reshape(row_shape)
                ).add_(self._eps)
                parameter_rows.sub_(
                    exp_avg_rows
                    / denominator
                    * step_size.reshape(row_shape)
                )
                parameter.index_copy_(0, target_page_ids, parameter_rows)
                state.step.index_copy_(0, target_page_ids, step_rows)
                state.exp_avg.index_copy_(0, target_page_ids, exp_avg_rows)
                state.exp_avg_sq.index_copy_(
                    0,
                    target_page_ids,
                    exp_avg_sq_rows,
                )


class FixedBatchedSlotAdapter(RequestSlotAdapter, Generic[BankT]):
    """One immutable-root page pool behind a typed capacity-first module."""

    __slots__ = (
        "_bank",
        "_layout_owner",
        "_layout",
        "_logical_capacity",
        "_resource",
        "_epoch",
        "_initial_parameters",
        "_initial_buffers",
        "_optimizer",
        "_parameter_count",
        "_buffer_count",
        "_roots",
        "_read_pins",
    )

    def __init__(
        self,
        bank: BankT,
        *,
        resource: StateResourceId,
        epoch: int,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if epoch < 1:
            raise ValueError("fixed batched state epoch must be positive")
        match bank:
            case FixedBatchedSlotModule() as layout_owner:
                initial_layout = layout_owner.fixed_batched_slot_layout()
            case _:
                raise TypeError(
                    "fixed batched adapter requires FixedBatchedSlotModule"
                )
        if initial_layout.capacity < 1 or not initial_layout.slot_parameters:
            raise ValueError("fixed batched slots require capacity and parameters")
        logical_capacity = initial_layout.capacity
        for tensor in (
            *initial_layout.slot_parameters,
            *initial_layout.slot_buffers,
        ):
            if tensor.ndim < 1 or int(tensor.shape[0]) != logical_capacity:
                raise ValueError("fixed batched tensors must be capacity-first")
        devices = {
            tensor.device
            for tensor in (
                *initial_layout.slot_parameters,
                *initial_layout.slot_buffers,
            )
        }
        if len(devices) != 1:
            raise ValueError("fixed batched tensors must share one device")
        device = initial_layout.slot_parameters[0].device
        initial_parameters = tuple(
            parameter.detach().clone()
            for parameter in initial_layout.slot_parameters
        )
        initial_buffers = tuple(
            buffer.detach().clone() for buffer in initial_layout.slot_buffers
        )
        physical_capacity = logical_capacity * 2
        physical_parameters = tuple(
            torch.nn.Parameter(
                torch.zeros(
                    (physical_capacity, *tuple(parameter.shape[1:])),
                    dtype=parameter.dtype,
                    device=device,
                ),
                requires_grad=parameter.requires_grad,
            )
            for parameter in initial_layout.slot_parameters
        )
        physical_buffers = tuple(
            torch.zeros(
                (physical_capacity, *tuple(buffer.shape[1:])),
                dtype=buffer.dtype,
                device=device,
            )
            for buffer in initial_layout.slot_buffers
        )
        physical_layout = FixedBatchedSlotLayout(
            physical_capacity,
            physical_parameters,
            physical_buffers,
        )
        layout_owner.install_fixed_batched_slot_layout(physical_layout)
        installed_layout = layout_owner.fixed_batched_slot_layout()
        if (
            installed_layout.capacity != physical_capacity
            or len(installed_layout.slot_parameters) != len(physical_parameters)
            or any(
                installed is not expected
                for installed, expected in zip(
                    installed_layout.slot_parameters,
                    physical_parameters,
                    strict=True,
                )
            )
            or len(installed_layout.slot_buffers) != len(physical_buffers)
            or any(
                installed is not expected
                for installed, expected in zip(
                    installed_layout.slot_buffers,
                    physical_buffers,
                    strict=True,
                )
            )
        ):
            raise RuntimeError("fixed batched module rejected its physical pools")
        super().__init__(capacity=logical_capacity, device=device)
        self._bank = bank
        self._layout_owner = layout_owner
        self._layout = installed_layout
        self._logical_capacity = logical_capacity
        self._resource = resource
        self._epoch = epoch
        self._initial_parameters = initial_parameters
        self._initial_buffers = initial_buffers
        self._optimizer = FixedBatchedAdamW(
            installed_layout,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self._parameter_count = len(installed_layout.slot_parameters)
        self._buffer_count = len(installed_layout.slot_buffers)
        pools = (
            *installed_layout.slot_parameters,
            *installed_layout.slot_buffers,
            *self._optimizer.state_tensors,
        )
        tensor_layout = FrozenTensorLayout(
            StateLayoutId(0),
            1,
            tuple(
                FrozenTensorField(
                    TensorFieldId(field_id),
                    tuple(int(size) for size in pool.shape[1:]),
                    pool.dtype,
                )
                for field_id, pool in enumerate(pools)
            ),
        )
        self._roots = PagedStateVersionArena(
            PageBundle.from_pools(tensor_layout, pools),
            bundle_epoch=epoch,
        )
        self._read_pins: dict[int, list[StateVersionRootPin]] = {}

    @property
    def bank(self) -> BankT:
        self._require_live()
        return self._bank

    @property
    def optimizer(self) -> FixedBatchedAdamW:
        self._require_live()
        return self._optimizer

    @property
    def version_roots(self) -> PagedStateVersionArena:
        self._require_live()
        return self._roots

    def prepare_admission(
        self,
        plan: RequestSlotAdmissionPlan,
    ) -> FixedBatchedAdmissionPreparation:
        self._require_live()
        slot_ids = tuple(admission.slot.slot_id for admission in plan.admissions)
        if any(slot_id >= self._logical_capacity for slot_id in slot_ids):
            raise RuntimeError("fixed batched admission exceeds logical capacity")
        self._roots.collect_completed()
        required_pages = len(plan.admissions)
        if self._roots.free_pages < required_pages:
            raise RuntimeError("fixed batched root capacity is exhausted")
        refs = tuple(
            StateVersionRef(
                StateVersionLineage(
                    self._resource,
                    plan.epoch,
                    admission.slot,
                ),
                0,
            )
            for admission in plan.admissions
        )
        return FixedBatchedAdmissionPreparation(
            plan.transaction_id,
            slot_ids,
            refs,
        )

    def commit_admission(
        self,
        plan: RequestSlotAdmissionPlan,
        preparation: FixedBatchedAdmissionPreparation,
    ) -> None:
        self._require_live()
        slot_ids = tuple(admission.slot.slot_id for admission in plan.admissions)
        refs = tuple(
            StateVersionRef(
                StateVersionLineage(
                    self._resource,
                    plan.epoch,
                    admission.slot,
                ),
                0,
            )
            for admission in plan.admissions
        )
        if (
            preparation.transaction_id != plan.transaction_id
            or preparation.slot_ids != slot_ids
            or preparation.refs != refs
        ):
            raise RuntimeError("fixed batched admission differs from its plan")
        try:
            for slot_id, ref in zip(slot_ids, refs, strict=True):
                write = self._roots.prepare_admission(ref)
                self._initialize_page(write, slot_id)
                self._roots.publish(write, self._record_readiness())
        except Exception:
            self.poison()
            raise

    def selected(
        self,
        selection: RequestStateSelectionBatch,
        versions: tuple[RequestStateVersion, ...],
    ) -> FixedBatchedSlotView[BankT]:
        self._require_live()
        refs = self._selection_refs(selection, versions)
        pin = self._pin_roots(refs)
        self._read_pins.setdefault(selection.selection_id, []).append(pin)
        page_by_lease = {
            lease: root.page_ids[0]
            for lease, root in zip(
                selection.leases,
                pin.roots,
                strict=True,
            )
        }
        return FixedBatchedSlotView(
            selection=selection,
            module=self._bank,
            row_slot_ids=torch.tensor(
                tuple(page_by_lease[row.lease] for row in selection.rows),
                dtype=torch.long,
                device=self.device,
            ),
            unique_slot_ids=torch.tensor(
                tuple(root.page_ids[0] for root in pin.roots),
                dtype=torch.long,
                device=self.device,
            ),
            valid_mask=selection.valid_mask,
            before_versions=tuple(version.version for version in versions),
        )

    def complete_reads_synchronous(
        self,
        selection: RequestStateSelectionBatch,
    ) -> None:
        self._require_live()
        if self.device.type != "cpu":
            raise RuntimeError("CUDA fixed batched reads require a completion event")
        if selection.selection_id not in self._read_pins:
            return
        pins = self._read_pins.pop(selection.selection_id)
        for pin in pins:
            self._roots.release_pin(pin, NoCompletionFence())

    def commit_fence(
        self,
        plan: RequestSlotPinPlan,
        fence: CompletionFence,
    ) -> None:
        super().commit_fence(plan, fence)
        if plan.selection.selection_id not in self._read_pins:
            return
        pins = self._read_pins.pop(plan.selection.selection_id)
        try:
            for pin in pins:
                self._roots.release_pin(pin, fence)
        except Exception:
            self.poison()
            raise

    def execute_transition(
        self,
        plan: RequestSlotTransitionPlan,
        losses: tuple[torch.Tensor, ...],
    ) -> None:
        self._require_live()
        if len(losses) != len(plan.transitions):
            raise ValueError("fixed batched losses must align with transition slots")
        if any(loss.numel() != 1 for loss in losses):
            raise ValueError("fixed batched transition losses must be scalar")
        self._validate_layout()
        self._roots.collect_completed()
        if self._roots.free_pages < len(plan.transitions):
            raise RuntimeError("fixed batched root capacity is exhausted")
        source_refs = tuple(
            StateVersionRef(
                StateVersionLineage(
                    self._resource,
                    plan.epoch,
                    effect.slot,
                ),
                effect.before_version,
            )
            for effect in plan.transitions
        )
        target_refs = tuple(
            StateVersionRef(source.lineage, effect.after_version)
            for source, effect in zip(
                source_refs,
                plan.transitions,
                strict=True,
            )
        )
        source_pin = self._pin_roots(source_refs)
        try:
            self._optimizer.zero_grad()
            with torch.inference_mode(False), torch.enable_grad():
                torch.autograd.backward(losses)
            for source, target, root in zip(
                source_refs,
                target_refs,
                source_pin.roots,
                strict=True,
            ):
                write = self._prepare_transition(source, target)
                self._update_page(write, root.page_ids[0])
                self._roots.publish(write, self._record_readiness())
            self._release_pin_after_current_work(source_pin)
            if self.device.type == "cpu":
                self.complete_reads_synchronous(plan.selection)
        except Exception:
            self.poison()
            raise

    def retire(self, receipt: StateEffectReceipt) -> None:
        self._require_live()
        if (
            receipt.resource != str(self._resource)
            or receipt.epoch != self._epoch
            or receipt.operation is not StateEffectOperation.RETIRE
            or self._read_pins
        ):
            raise RuntimeError("fixed batched retirement has active or foreign state")
        try:
            for effect in receipt.slots:
                self._roots.retire(
                    StateVersionRef(
                        StateVersionLineage(
                            self._resource,
                            receipt.epoch,
                            effect.slot,
                        ),
                        effect.after_version,
                    )
                )
        except Exception:
            self.poison()
            raise

    def replace_epoch(self, epoch: int) -> None:
        self._require_live()
        self._roots.collect_completed()
        if self._read_pins or self._roots.free_pages != self._roots.bundle.capacity:
            raise RuntimeError("cannot replace fixed batched epoch with live roots")
        self._roots = PagedStateVersionArena(
            self._roots.bundle,
            bundle_epoch=epoch,
        )
        self._epoch = epoch

    def poison(self) -> None:
        super().poison()
        self._roots.poison()

    def _selection_refs(
        self,
        selection: RequestStateSelectionBatch,
        versions: tuple[RequestStateVersion, ...],
    ) -> tuple[StateVersionRef, ...]:
        if len(versions) != len(selection.leases) or any(
            version.key != lease.key or version.slot != lease.slot
            for version, lease in zip(versions, selection.leases, strict=True)
        ):
            raise RuntimeError("fixed batched versions differ from selection leases")
        return tuple(
            StateVersionRef(
                StateVersionLineage(
                    self._resource,
                    selection.epoch,
                    lease.slot,
                ),
                version.version,
            )
            for lease, version in zip(
                selection.leases,
                versions,
                strict=True,
            )
        )

    def _pin_roots(
        self,
        refs: tuple[StateVersionRef, ...],
    ) -> StateVersionRootPin:
        if self.device.type == "cpu":
            return self._roots.pin_exact_synchronous(refs)
        return self._roots.pin_exact_cuda(
            refs,
            torch.cuda.current_stream(self.device),
        )

    def _prepare_transition(
        self,
        source: StateVersionRef,
        target: StateVersionRef,
    ) -> TransitionPageWrite:
        if self.device.type == "cpu":
            return self._roots.prepare_transition_synchronous(source, target)
        return self._roots.prepare_transition_cuda(
            source,
            target,
            torch.cuda.current_stream(self.device),
        )

    def _record_readiness(self) -> CompletionFence:
        if self.device.type == "cpu":
            return NoCompletionFence()
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        return CudaCompletionFence(event)

    def _release_pin_after_current_work(
        self,
        pin: StateVersionRootPin,
    ) -> None:
        self._roots.release_pin(pin, self._record_readiness())

    def _initialize_page(
        self,
        write: AdmissionPageWrite,
        logical_slot_id: int,
    ) -> None:
        (target_page,) = write.target_pages
        with torch.no_grad():
            for target, initial in zip(
                target_page[: self._parameter_count],
                self._initial_parameters,
                strict=True,
            ):
                target.copy_(initial[logical_slot_id])
            buffer_start = self._parameter_count
            buffer_end = buffer_start + self._buffer_count
            for target, initial in zip(
                target_page[buffer_start:buffer_end],
                self._initial_buffers,
                strict=True,
            ):
                target.copy_(initial[logical_slot_id])
            for target in target_page[buffer_end:]:
                target.zero_()

    def _update_page(
        self,
        write: TransitionPageWrite,
        source_page_id: int,
    ) -> None:
        (target_page_id,) = write.target_page_ids
        source_ids = torch.tensor(
            (source_page_id,),
            dtype=torch.long,
            device=self.device,
        )
        target_ids = torch.tensor(
            (target_page_id,),
            dtype=torch.long,
            device=self.device,
        )
        self._optimizer.step(source_ids, target_ids)
        with torch.no_grad():
            for pool in self._layout.slot_buffers:
                pool.index_copy_(0, target_ids, pool.index_select(0, source_ids))

    def _validate_layout(self) -> None:
        current = self._layout_owner.fixed_batched_slot_layout()
        if (
            current.capacity != self._layout.capacity
            or len(current.slot_parameters) != len(self._layout.slot_parameters)
            or any(
                current_parameter is not initial_parameter
                for current_parameter, initial_parameter in zip(
                    current.slot_parameters,
                    self._layout.slot_parameters,
                    strict=True,
                )
            )
            or len(current.slot_buffers) != len(self._layout.slot_buffers)
            or any(
                current_buffer is not initial_buffer
                for current_buffer, initial_buffer in zip(
                    current.slot_buffers,
                    self._layout.slot_buffers,
                    strict=True,
                )
            )
        ):
            raise RuntimeError("fixed batched slot layout changed")


__all__ = [
    "FixedBatchedAdamW",
    "FixedBatchedAdamWState",
    "FixedBatchedAdmissionPreparation",
    "FixedBatchedSlotAdapter",
    "FixedBatchedSlotLayout",
    "FixedBatchedSlotModule",
    "FixedBatchedSlotView",
]
