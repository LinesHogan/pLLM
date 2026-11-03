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
        field_ids = {self.parameter, self.gradient, self.step, self.exp_avg, self.exp_avg_sq}
        assert len(field_ids) == 5 and min(field_ids) >= 0


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
            raise ValueError(
                f"AdamW needs finite values with lr > 0, eps > 0, weight_decay >= 0 and betas in [0, 1); got {self}"
            )


def apply_adamw(
    roots: Sequence[tuple[torch.Tensor, ...]],
    fields: AdamWFields,
    config: AdamWConfig,
    gradients: tuple[torch.Tensor, ...],
) -> None:
    """Apply the bound optimizer to validated page views and matching gradients."""
    assert roots and len(roots) == len(gradients)
    assert all(
        (gradient.shape, gradient.device) == (root[fields.parameter].shape, root[fields.parameter].device)
        for root, gradient in zip(roots, gradients, strict=True)
    )
    with torch.no_grad():
        for root, gradient in zip(roots, gradients, strict=True):
            root[fields.gradient].copy_(gradient)
        cuda = roots[0][fields.parameter].is_cuda
        parameters = [root[fields.parameter].to(root[fields.exp_avg].dtype) for root in roots]
        adamw(
            params=parameters,
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

        for root, parameter in zip(roots, parameters, strict=True):
            if parameter is not root[fields.parameter]:
                root[fields.parameter].copy_(parameter)


@triton.jit
def _advance_steps(
    steps: torch.Tensor,
    slots: torch.Tensor,
    sources: torch.Tensor,
    corrections: torch.Tensor,
    count: tl.constexpr,
    stride: tl.constexpr,
    lr: float,
    alpha1: float,
    alpha2: float,
) -> None:
    state = tl.program_id(0)
    slot = tl.load(slots + 2 * count + state)
    source = tl.load(sources + 2 * count + state)
    value = tl.load(steps + source * stride, slot >= 0, 0) + 1
    tl.store(steps + slot * stride, value, slot >= 0)
    # The bias corrections 1 - beta**step, in FP64 from alpha = 1 - beta: an FP32 beta near 1 (0.999 rounds by
    # 1.3e-8) would put a 1e-5 relative error into 1 - beta2**step.
    steps64 = tl.cast(value, tl.float64)
    correction1 = -libdevice.expm1(steps64 * libdevice.log1p(-tl.cast(alpha1, tl.float64)))
    correction2 = -libdevice.expm1(steps64 * libdevice.log1p(-tl.cast(alpha2, tl.float64)))
    tl.store(corrections + state * 2, (-tl.cast(lr, tl.float64) / correction1).to(tl.float32))
    tl.store(corrections + state * 2 + 1, tl.sqrt(correction2).to(tl.float32))


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
    assert gradients.shape == (slots.shape[1], pools[0][0].numel()) and gradients.dtype == torch.float32
    assert gradients.device == slots.device and gradients.is_contiguous() and not gradients.requires_grad
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
    _advance_steps[(count,)](step, slots, sources, corrections, count, step.stride(0), lr, 1 - beta1, 1 - beta2)
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
