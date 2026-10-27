"""Independent AdamW tensor update over private PageBundle fields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import torch
import triton
import triton.language as tl
from torch.optim.adamw import adamw
from triton.language.extra.cuda import libdevice

from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.kernels.page_copy import no_allocation


@dataclass(frozen=True, slots=True)
class AdamWFields:
    parameter: LayoutFieldId
    gradient: LayoutFieldId
    step: LayoutFieldId
    exp_avg: LayoutFieldId
    exp_avg_sq: LayoutFieldId

    def __post_init__(self) -> None:
        field_ids = (
            int(self.parameter),
            int(self.gradient),
            int(self.step),
            int(self.exp_avg),
            int(self.exp_avg_sq),
        )
        if any(field_id < 0 for field_id in field_ids) or len(set(field_ids)) != len(field_ids):
            raise ValueError("dense-page AdamW fields must be distinct IDs")


@dataclass(frozen=True, slots=True)
class AdamWConfig:
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        beta_one, beta_two = self.betas
        if (
            not all(isfinite(value) for value in (self.lr, beta_one, beta_two, self.eps, self.weight_decay))
            or self.lr <= 0.0
            or not 0.0 <= beta_one < 1.0
            or not 0.0 <= beta_two < 1.0
            or self.eps <= 0.0
            or self.weight_decay < 0.0
        ):
            raise ValueError("dense-page AdamW configuration is invalid")


def apply_adamw(
    roots: Sequence[tuple[torch.Tensor, ...]],
    fields: AdamWFields,
    config: AdamWConfig,
    gradients: tuple[torch.Tensor, ...],
) -> None:
    """Apply the bound optimizer to validated page views and matching gradients."""
    # PTTS and tensor input binding already validate page fields; gradients enter here from math.
    if (
        not roots
        or len(roots) != len(gradients)
        or any(
            (gradient.shape, gradient.dtype, gradient.device)
            != (root[fields.parameter].shape, root[fields.parameter].dtype, root[fields.parameter].device)
            for root, gradient in zip(roots, gradients, strict=True)
        )
    ):
        raise ValueError("AdamW gradients must match the parameter pages")
    with torch.no_grad():
        for root, gradient in zip(roots, gradients, strict=True):
            root[fields.gradient].copy_(gradient)
        cuda = roots[0][fields.parameter].is_cuda
        adamw(
            params=[root[fields.parameter] for root in roots],
            grads=[root[fields.gradient] for root in roots],
            exp_avgs=[root[fields.exp_avg] for root in roots],
            exp_avg_sqs=[root[fields.exp_avg_sq] for root in roots],
            max_exp_avg_sqs=[],
            state_steps=[root[fields.step] for root in roots],
            foreach=cuda,
            capturable=cuda,
            differentiable=False,
            fused=False,
            amsgrad=False,
            beta1=config.betas[0],
            beta2=config.betas[1],
            lr=config.lr,
            weight_decay=config.weight_decay,
            eps=config.eps,
            maximize=False,
        )


@triton.jit
def _advance_steps(
    steps: torch.Tensor,
    slots: torch.Tensor,
    sources: torch.Tensor,
    corrections: torch.Tensor,
    count: tl.constexpr,
    stride: tl.constexpr,
    lr: float,
    beta1: float,
    beta2: float,
) -> None:
    state = tl.program_id(0)
    slot = tl.load(slots + 2 * count + state)
    source = tl.load(sources + 2 * count + state)
    value = tl.load(steps + source * stride, slot >= 0, 0) + 1
    tl.store(steps + slot * stride, value, slot >= 0)
    power1 = libdevice.pow(tl.cast(beta1, tl.float64), tl.cast(value, tl.float64)).to(tl.float32)
    power2 = libdevice.pow(tl.cast(beta2, tl.float64), tl.cast(value, tl.float64)).to(tl.float32)
    tl.store(corrections + state * 2, tl.div_rn(1.0, tl.div_rn(power1 - 1, lr)))
    tl.store(corrections + state * 2 + 1, tl.sqrt(1 - power2))


@triton.jit
def _paged_adamw(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
    sources: torch.Tensor,
    corrections: torch.Tensor,
    count: tl.constexpr,
    width: tl.constexpr,
    p_stride: tl.constexpr,
    g_stride: tl.constexpr,
    m_stride: tl.constexpr,
    v_stride: tl.constexpr,
    beta2: float,
    alpha1: float,
    alpha2: float,
    eps: float,
    decay: float,
) -> None:
    state = tl.program_id(0) // tl.cdiv(width, 256)
    column = (tl.program_id(0) % tl.cdiv(width, 256)) * 256 + tl.arange(0, 256)
    p = tl.load(slots + state)
    g = tl.load(slots + count + state)
    m = tl.load(slots + 3 * count + state)
    v = tl.load(slots + 4 * count + state)
    valid = (p >= 0) & (column < width)
    grad = tl.load(values + state * width + column, valid, 0)
    source_p = tl.load(sources + state)
    source_m = tl.load(sources + 3 * count + state)
    source_v = tl.load(sources + 4 * count + state)
    first = tl.load(exp_avg + source_m * m_stride + column, valid, 0)
    second = tl.load(exp_avg_sq + source_v * v_stride + column, valid, 0)
    first = tl.where(alpha1 < 0.5, first + (grad - first) * alpha1, grad - (grad - first) * (1 - alpha1))
    second = libdevice.mul_rn(second, beta2) + alpha2 * grad * grad
    step_size = tl.load(corrections + state * 2)
    correction2 = tl.load(corrections + state * 2 + 1)
    denominator = tl.div_rn(tl.div_rn(tl.sqrt(second), correction2) + eps, step_size)
    value = libdevice.mul_rn(tl.load(parameter + source_p * p_stride + column, valid, 0), decay)
    value = value + tl.div_rn(first, denominator)
    tl.store(parameter + p * p_stride + column, value, valid)
    tl.store(gradient + g * g_stride + column, grad, valid)
    tl.store(exp_avg + m * m_stride + column, first, valid)
    tl.store(exp_avg_sq + v * v_stride + column, second, valid)


def paged_adamw(
    pools: tuple[torch.Tensor, ...],
    slots: torch.Tensor,
    gradients: torch.Tensor,
    config: AdamWConfig,
    sources: torch.Tensor | None = None,
) -> None:
    """Five original FP32 fields, authorized slot prefixes, no field packing."""
    if (
        gradients.shape != (slots.shape[1], pools[0][0].numel())
        or gradients.dtype != torch.float32
        or gradients.device != slots.device
        or not gradients.is_contiguous()
        or gradients.requires_grad
    ):
        raise ValueError("paged AdamW gradients must match the selected parameter pages")
    sources = slots if sources is None else sources
    corrections = gradients.new_empty((len(gradients), 2))
    _paged_update(
        list(pools), slots, gradients, sources, corrections, [config.lr, *config.betas, config.eps, config.weight_decay]
    )


@torch.library.custom_op("pllm::paged_adamw", mutates_args={"pools", "corrections"})
def _paged_update(
    pools: list[torch.Tensor],
    slots: torch.Tensor,
    gradients: torch.Tensor,
    sources: torch.Tensor,
    corrections: torch.Tensor,
    hyperparameters: list[float],
) -> None:
    parameter, gradient, step, first, second = pools
    lr, beta1, beta2, eps, decay = hyperparameters
    count, width = gradients.shape
    # Increment in a separate launch: no tile may observe another tile's step write.
    _advance_steps[(count,)](step, slots, sources, corrections, count, step.stride(0), lr, beta1, beta2)
    _paged_adamw[(count * ((width + 255) // 256),)](
        parameter,
        gradient,
        first,
        second,
        slots,
        gradients,
        sources,
        corrections,
        count,
        width,
        parameter.stride(0),
        gradient.stride(0),
        first.stride(0),
        second.stride(0),
        beta2,
        1 - beta1,
        1 - beta2,
        eps,
        1 - lr * decay,
    )


_paged_update.register_fake(no_allocation)  # pyright: ignore[reportUnknownMemberType]
