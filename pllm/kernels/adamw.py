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
    ids = (fields.parameter, fields.gradient, fields.step, fields.exp_avg, fields.exp_avg_sq)
    if len(roots) != len(gradients) or not roots:
        raise ValueError("AdamW gradients must align with private roots")
    for root, gradient in zip(roots, gradients, strict=True):
        if any(field >= len(root) for field in ids):
            raise ValueError("AdamW field is outside the signed layout")
        parameter = root[fields.parameter]
        if (
            any(
                root[field].shape != parameter.shape or root[field].dtype != parameter.dtype
                for field in (fields.gradient, fields.exp_avg, fields.exp_avg_sq)
            )
            or root[fields.step].shape != ()
            or root[fields.step].dtype != torch.float32
            or gradient.shape != parameter.shape
            or gradient.dtype != parameter.dtype
            or gradient.device != parameter.device
        ):
            raise ValueError("AdamW fields or gradient differ from the signed layout")
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
    corrections: torch.Tensor,
    count: tl.constexpr,
    stride: tl.constexpr,
    lr: float,
    beta1: float,
    beta2: float,
) -> None:
    state = tl.program_id(0)
    slot = tl.load(slots + 2 * count + state)
    value = tl.load(steps + slot * stride, slot >= 0, 0) + 1
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
    state = tl.program_id(0)
    column = tl.program_id(1) * 256 + tl.arange(0, 256)
    p = tl.load(slots + state)
    g = tl.load(slots + count + state)
    m = tl.load(slots + 3 * count + state)
    v = tl.load(slots + 4 * count + state)
    valid = (p >= 0) & (column < width)
    grad = tl.load(values + state * width + column, valid, 0)
    first = tl.load(exp_avg + m * m_stride + column, valid, 0)
    second = tl.load(exp_avg_sq + v * v_stride + column, valid, 0)
    first = tl.where(alpha1 < 0.5, first + (grad - first) * alpha1, grad - (grad - first) * (1 - alpha1))
    second = libdevice.mul_rn(second, beta2) + alpha2 * grad * grad
    step_size = tl.load(corrections + state * 2)
    correction2 = tl.load(corrections + state * 2 + 1)
    denominator = tl.div_rn(tl.div_rn(tl.sqrt(second), correction2) + eps, step_size)
    value = libdevice.mul_rn(tl.load(parameter + p * p_stride + column, valid, 0), decay)
    value = value + tl.div_rn(first, denominator)
    tl.store(parameter + p * p_stride + column, value, valid)
    tl.store(gradient + g * g_stride + column, grad, valid)
    tl.store(exp_avg + m * m_stride + column, first, valid)
    tl.store(exp_avg_sq + v * v_stride + column, second, valid)


def paged_adamw(
    pools: tuple[torch.Tensor, ...], slots: torch.Tensor, gradients: torch.Tensor, config: AdamWConfig
) -> None:
    """Five original FP32 fields, authorized slot prefixes, no field packing."""
    if (
        len(pools) != 5
        or slots.ndim != 2
        or slots.shape[0] != 5
        or slots.dtype != torch.long
        or not slots.is_contiguous()
    ):
        raise ValueError("paged AdamW requires five original field-slot rows")
    parameter, gradient, step, first, second = pools
    if (
        parameter.ndim < 2
        or any(pool.dtype != torch.float32 or pool.device != slots.device for pool in pools)
        or any(pool.shape != parameter.shape for pool in (gradient, first, second))
        or step.shape != parameter.shape[:1]
        or gradients.shape != (slots.shape[1], parameter[0].numel())
        or gradients.dtype != torch.float32
        or gradients.device != slots.device
        or not gradients.is_contiguous()
        or gradients.requires_grad
    ):
        raise ValueError("paged AdamW fields or gradients differ from the signed layout")
    count, width = gradients.shape
    corrections = gradients.new_empty((count, 2))
    # Increment in a separate launch: no tile may observe another tile's step write.
    _advance_steps[(count,)](
        step, slots, corrections, count, step.stride(0), config.lr, config.betas[0], config.betas[1]
    )
    _paged_adamw[(count, (width + 255) // 256)](
        parameter,
        gradient,
        first,
        second,
        slots,
        gradients,
        corrections,
        count,
        width,
        parameter.stride(0),
        gradient.stride(0),
        first.stride(0),
        second.stride(0),
        config.betas[1],
        1 - config.betas[0],
        1 - config.betas[1],
        config.eps,
        1 - config.lr * config.weight_decay,
    )
