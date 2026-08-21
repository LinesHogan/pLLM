#!/usr/bin/env python3
"""Functional optimizer transitions for Program state banks.

Pure-tensor implementations of the v1-supported optimizers, mirroring the
single-tensor (non-foreach) ``torch.optim`` reference semantics exactly.
Each function maps ``(params, grads, state) -> (new_params, new_state)``;
nothing mutates in place, so state banks can version and fence updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from tllm.workgraph.lowering import WorkGraphError

TensorMap = dict[str, torch.Tensor]
OptState = dict[str, dict[str, torch.Tensor]]


@dataclass(frozen=True, slots=True)
class SGDScalars:
    lr: float
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False


@dataclass(frozen=True, slots=True)
class AdamScalars:
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    decoupled: bool = False  # True for AdamW semantics


def sgd_step(
    params: Mapping[str, torch.Tensor],
    grads: Mapping[str, torch.Tensor],
    state: OptState,
    scalars: SGDScalars,
) -> tuple[TensorMap, OptState]:
    """One functional SGD step (torch.optim.SGD single-tensor semantics)."""
    new_params: TensorMap = {}
    new_state: OptState = {}
    for name, param in params.items():
        grad = grads[name]
        if scalars.weight_decay != 0.0:
            grad = grad.add(param, alpha=scalars.weight_decay)
        if scalars.momentum != 0.0:
            prior = state.get(name, {}).get("momentum_buffer")
            buffer = (
                grad.clone()
                if prior is None
                else prior.mul(scalars.momentum).add(grad, alpha=1.0 - scalars.dampening)
            )
            new_state[name] = {"momentum_buffer": buffer}
            grad = grad.add(buffer, alpha=scalars.momentum) if scalars.nesterov else buffer
        new_params[name] = param.add(grad, alpha=-scalars.lr)
    return new_params, new_state


def adam_step(
    params: Mapping[str, torch.Tensor],
    grads: Mapping[str, torch.Tensor],
    state: OptState,
    scalars: AdamScalars,
    *,
    step: int,
) -> tuple[TensorMap, OptState]:
    """One functional Adam/AdamW step (single-tensor reference semantics)."""
    if step < 1:
        raise ValueError("adam_step requires step >= 1")
    beta1, beta2 = scalars.betas
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    step_size = scalars.lr / bias_correction1
    new_params: TensorMap = {}
    new_state: OptState = {}
    for name, param in params.items():
        grad = grads[name]
        prior = state.get(name, {})
        exp_avg = prior.get("exp_avg", torch.zeros_like(param))
        exp_avg_sq = prior.get("exp_avg_sq", torch.zeros_like(param))
        if scalars.weight_decay != 0.0 and not scalars.decoupled:
            grad = grad.add(param, alpha=scalars.weight_decay)
        exp_avg = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
        exp_avg_sq = exp_avg_sq.mul(beta2).add(grad * grad, alpha=1.0 - beta2)
        new_state[name] = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
        updated = param
        if scalars.weight_decay != 0.0 and scalars.decoupled:
            updated = param.mul(1.0 - scalars.lr * scalars.weight_decay)
        denom = exp_avg_sq.sqrt().div(bias_correction2**0.5).add(scalars.eps)
        new_params[name] = updated.addcdiv(exp_avg, denom, value=-step_size)
    return new_params, new_state


def validate_optimizer(optimizer: torch.optim.Optimizer) -> SGDScalars | AdamScalars:
    """Extract fixed group scalars from a supported optimizer, or PG160.

    v1 supports exactly one parameter group with fixed scalars, no
    maximize/amsgrad, and no sparse gradients.
    """
    label = f"optimizer {type(optimizer).__name__}"
    if len(optimizer.param_groups) != 1:
        raise WorkGraphError("PG160", f"{label}: v1 requires exactly one parameter group")
    group = optimizer.param_groups[0]
    if group.get("maximize", False):
        raise WorkGraphError("PG160", f"{label}: maximize=True is unsupported")
    if isinstance(optimizer, torch.optim.SGD):
        return SGDScalars(
            lr=float(group["lr"]),
            momentum=float(group["momentum"]),
            dampening=float(group["dampening"]),
            weight_decay=float(group["weight_decay"]),
            nesterov=bool(group["nesterov"]),
        )
    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        if group.get("amsgrad", False):
            raise WorkGraphError("PG160", f"{label}: amsgrad=True is unsupported")
        betas = group["betas"]
        return AdamScalars(
            lr=float(group["lr"]),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(group["eps"]),
            weight_decay=float(group["weight_decay"]),
            decoupled=isinstance(optimizer, torch.optim.AdamW),
        )
    raise WorkGraphError(
        "PG160",
        f"unsupported optimizer {type(optimizer).__name__}; supported: SGD, Adam, AdamW",
    )


def functional_step(
    scalars: SGDScalars | AdamScalars,
    params: Mapping[str, torch.Tensor],
    grads: Mapping[str, torch.Tensor],
    state: OptState,
    *,
    step: int,
) -> tuple[TensorMap, OptState]:
    if isinstance(scalars, SGDScalars):
        return sgd_step(params, grads, state, scalars)
    return adam_step(params, grads, state, scalars, step=step)


def adam_step_inplace_(
    params: Mapping[str, torch.Tensor],
    grads: Mapping[str, torch.Tensor],
    state: OptState,
    scalars: AdamScalars,
    *,
    step: torch.Tensor,
    beta1: torch.Tensor,
    beta2: torch.Tensor,
) -> None:
    """In-place tensorized Adam/AdamW step for CUDA-graph-captured updates.

    Same update math as :func:`adam_step` but with the step-dependent scalars
    computed on device from 0-dim tensors, so the whole step is capturable and
    replayable without host involvement. ``step``/``beta1``/``beta2`` are
    float32 device scalars; ``step`` holds the 1-based step count. Operates in
    place on the persistent slot buffers (params and exp_avg/exp_avg_sq).
    """
    beta1_f, beta2_f = scalars.betas
    one = beta1.new_ones(())
    bias_correction1 = one - torch.pow(beta1, step)
    bias_correction2_sqrt = torch.sqrt(one - torch.pow(beta2, step))
    step_size = scalars.lr / bias_correction1
    decay = scalars.lr * scalars.weight_decay
    for name, param in params.items():
        grad = grads[name]
        if scalars.weight_decay != 0.0 and not scalars.decoupled:
            grad = grad.add(param, alpha=scalars.weight_decay)
        buffers = state[name]
        exp_avg = buffers["exp_avg"]
        exp_avg_sq = buffers["exp_avg_sq"]
        exp_avg.mul_(beta1_f).add_(grad, alpha=1.0 - beta1_f)
        exp_avg_sq.mul_(beta2_f).addcmul_(grad, grad, value=1.0 - beta2_f)
        denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(scalars.eps)
        if scalars.weight_decay != 0.0 and scalars.decoupled:
            param.mul_(1.0 - decay)
        param.sub_(exp_avg.div(denom).mul_(step_size))


__all__ = [
    "AdamScalars",
    "OptState",
    "SGDScalars",
    "TensorMap",
    "adam_step",
    "adam_step_inplace_",
    "functional_step",
    "sgd_step",
    "validate_optimizer",
]
