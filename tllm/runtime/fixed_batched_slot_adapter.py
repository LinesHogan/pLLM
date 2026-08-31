"""Fixed-address capacity-first physical storage for request slots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from tllm.contracts.request_state import RequestStateSelectionBatch
from tllm.runtime.request_slot_adapter import RequestSlotAdapter
from tllm.runtime.request_slot_arena import (
    RequestSlotAdmissionPlan,
    RequestSlotTransitionPlan,
)


@dataclass(frozen=True, slots=True)
class FixedBatchedSlotLayout:
    """Explicit capacity-first tensors owned by one fixed bank module."""

    capacity: int
    slot_parameters: tuple[torch.nn.Parameter, ...]
    slot_buffers: tuple[torch.Tensor, ...]


class FixedBatchedSlotModule(torch.nn.Module):
    """Nominal base for modules whose mutable tensors are slot-major."""

    def fixed_batched_slot_layout(self) -> FixedBatchedSlotLayout:
        raise NotImplementedError


BankT = TypeVar("BankT", bound=torch.nn.Module)


@dataclass(frozen=True, slots=True)
class FixedBatchedAdmissionPreparation:
    transaction_id: int
    slot_ids: tuple[int, ...]


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
    """AdamW whose moment and step state is independently indexed by slot."""

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
                    (layout.capacity,), dtype=torch.float32, device=parameter.device
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

    def reset_slots(self, slot_ids: torch.Tensor) -> None:
        with torch.no_grad():
            for state in self._states:
                state.step.index_fill_(0, slot_ids, 0)
                state.exp_avg.index_fill_(0, slot_ids, 0.0)
                state.exp_avg_sq.index_fill_(0, slot_ids, 0.0)

    def zero_grad(self) -> None:
        for gradient in self._gradients:
            gradient.zero_()

    def step(self, slot_ids: torch.Tensor) -> None:
        with torch.no_grad():
            for parameter, state, gradient in zip(
                self._parameters,
                self._states,
                self._gradients,
                strict=True,
            ):
                parameter_rows = parameter.index_select(0, slot_ids)
                gradient_rows = gradient.index_select(0, slot_ids)
                exp_avg_rows = state.exp_avg.index_select(0, slot_ids)
                exp_avg_sq_rows = state.exp_avg_sq.index_select(0, slot_ids)
                step_rows = state.step.index_select(0, slot_ids).add_(1)
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
                row_shape = (int(slot_ids.shape[0]),) + (1,) * (
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
                parameter.index_copy_(0, slot_ids, parameter_rows)
                state.step.index_copy_(0, slot_ids, step_rows)
                state.exp_avg.index_copy_(0, slot_ids, exp_avg_rows)
                state.exp_avg_sq.index_copy_(0, slot_ids, exp_avg_sq_rows)


class FixedBatchedSlotAdapter(RequestSlotAdapter, Generic[BankT]):
    """One bank and one per-slot AdamW state without logical slot ownership."""

    __slots__ = (
        "_bank",
        "_layout_owner",
        "_layout",
        "_initial_parameters",
        "_initial_buffers",
        "_optimizer",
    )

    def __init__(
        self,
        bank: BankT,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        match bank:
            case FixedBatchedSlotModule() as layout_owner:
                layout = layout_owner.fixed_batched_slot_layout()
            case _:
                raise TypeError(
                    "fixed batched adapter requires FixedBatchedSlotModule"
                )
        if layout.capacity < 1 or not layout.slot_parameters:
            raise ValueError("fixed batched slots require capacity and parameters")
        for tensor in (*layout.slot_parameters, *layout.slot_buffers):
            if tensor.ndim < 1 or tensor.shape[0] != layout.capacity:
                raise ValueError("fixed batched tensors must be capacity-first")
        devices = {tensor.device for tensor in (
            *layout.slot_parameters,
            *layout.slot_buffers,
        )}
        if len(devices) != 1:
            raise ValueError("fixed batched tensors must share one device")
        device = layout.slot_parameters[0].device
        super().__init__(capacity=layout.capacity, device=device)
        self._bank = bank
        self._layout_owner = layout_owner
        self._layout = layout
        self._initial_parameters = tuple(
            parameter.detach().clone() for parameter in layout.slot_parameters
        )
        self._initial_buffers = tuple(
            buffer.detach().clone() for buffer in layout.slot_buffers
        )
        self._optimizer = FixedBatchedAdamW(
            layout,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    def selected(
        self,
        selection: RequestStateSelectionBatch,
    ) -> FixedBatchedSlotView[BankT]:
        self._require_live()
        return FixedBatchedSlotView(
            selection=selection,
            module=self._bank,
            row_slot_ids=selection.device_slot_ids,
            unique_slot_ids=selection.device_lease_slot_ids,
            valid_mask=selection.valid_mask,
            before_versions=selection.before_versions,
        )

    @property
    def bank(self) -> BankT:
        self._require_live()
        return self._bank

    @property
    def optimizer(self) -> FixedBatchedAdamW:
        self._require_live()
        return self._optimizer

    def prepare_admission(
        self,
        plan: RequestSlotAdmissionPlan,
    ) -> FixedBatchedAdmissionPreparation:
        self._require_live()
        slot_ids = tuple(admission.slot.slot_id for admission in plan.admissions)
        if any(slot_id >= self._layout.capacity for slot_id in slot_ids):
            raise RuntimeError("fixed batched admission exceeds bank capacity")
        return FixedBatchedAdmissionPreparation(plan.transaction_id, slot_ids)

    def commit_admission(
        self,
        plan: RequestSlotAdmissionPlan,
        preparation: FixedBatchedAdmissionPreparation,
    ) -> None:
        self._require_live()
        slot_ids = tuple(admission.slot.slot_id for admission in plan.admissions)
        if (
            preparation.transaction_id != plan.transaction_id
            or preparation.slot_ids != slot_ids
        ):
            raise RuntimeError("fixed batched admission differs from its plan")
        device_slot_ids = torch.tensor(
            slot_ids,
            dtype=torch.long,
            device=self.device,
        )
        self._validate_layout()
        try:
            with torch.no_grad():
                for parameter, initial in zip(
                    self._layout.slot_parameters,
                    self._initial_parameters,
                    strict=True,
                ):
                    parameter.index_copy_(
                        0,
                        device_slot_ids,
                        initial.index_select(0, device_slot_ids),
                    )
                for buffer, initial in zip(
                    self._layout.slot_buffers,
                    self._initial_buffers,
                    strict=True,
                ):
                    buffer.index_copy_(
                        0,
                        device_slot_ids,
                        initial.index_select(0, device_slot_ids),
                    )
            self._optimizer.reset_slots(device_slot_ids)
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
        slot_ids = torch.tensor(
            tuple(effect.slot.slot_id for effect in plan.transitions),
            dtype=torch.long,
            device=self.device,
        )
        try:
            self._optimizer.zero_grad()
            with torch.inference_mode(False), torch.enable_grad():
                torch.autograd.backward(losses)
            self._optimizer.step(slot_ids)
        except Exception:
            self.poison()
            raise

    def _validate_layout(self) -> None:
        current = self._layout_owner.fixed_batched_slot_layout()
        if (
            current.capacity != self._layout.capacity
            or len(current.slot_parameters)
            != len(self._layout.slot_parameters)
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
