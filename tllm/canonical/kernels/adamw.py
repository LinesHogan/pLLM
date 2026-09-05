"""Independent AdamW tensor update over private PageBundle fields."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch.optim.adamw import adamw

from tllm.contracts.tensor_layout import LayoutFieldId


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
        if any(field_id < 0 for field_id in field_ids) or len(
            set(field_ids)
        ) != len(field_ids):
            raise ValueError("dense-page AdamW fields must be distinct IDs")


@dataclass(frozen=True, slots=True)
class AdamWConfig:
    lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float

    def __post_init__(self) -> None:
        if type(self.betas) is not tuple or len(self.betas) != 2:
            raise ValueError("dense-page AdamW betas must be an exact pair")
        beta_one, beta_two = self.betas
        if (
            type(self.lr) is not float
            or type(beta_one) is not float
            or type(beta_two) is not float
            or type(self.eps) is not float
            or type(self.weight_decay) is not float
            or not all(
                isfinite(value)
                for value in (
                    self.lr,
                    beta_one,
                    beta_two,
                    self.eps,
                    self.weight_decay,
                )
            )
            or self.lr <= 0.0
            or not 0.0 <= beta_one < 1.0
            or not 0.0 <= beta_two < 1.0
            or self.eps <= 0.0
            or self.weight_decay < 0.0
        ):
            raise ValueError("dense-page AdamW configuration is invalid")


def apply_adamw(
    roots: tuple[tuple[torch.Tensor, ...], ...],
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
        if (any(root[field].shape != parameter.shape or root[field].dtype != parameter.dtype
                for field in (fields.gradient, fields.exp_avg, fields.exp_avg_sq))
            or root[fields.step].shape != () or root[fields.step].dtype != torch.float32
            or gradient.shape != parameter.shape or gradient.dtype != parameter.dtype
            or gradient.device != parameter.device):
            raise ValueError("AdamW fields or gradient differ from the signed layout")
    with torch.no_grad():
        for root, gradient in zip(roots, gradients, strict=True):
            root[fields.gradient].copy_(gradient)
        if roots[0][fields.parameter].device.type == "cpu":
            adamw(
                params=[root[fields.parameter] for root in roots],
                grads=[root[fields.gradient] for root in roots],
                exp_avgs=[root[fields.exp_avg] for root in roots],
                exp_avg_sqs=[root[fields.exp_avg_sq] for root in roots],
                max_exp_avg_sqs=[], state_steps=[root[fields.step] for root in roots],
                foreach=False, capturable=False, differentiable=False, fused=False,
                amsgrad=False, beta1=config.betas[0], beta2=config.betas[1],
                lr=config.lr, weight_decay=config.weight_decay, eps=config.eps, maximize=False,
            )
            return
        beta_one, beta_two = config.betas
        for root, gradient in zip(roots, gradients, strict=True):
            parameter, step = root[fields.parameter], root[fields.step]
            exp_avg, exp_avg_sq = root[fields.exp_avg], root[fields.exp_avg_sq]
            step.add_(1)
            exp_avg.mul_(beta_one).add_(gradient, alpha=1.0 - beta_one)
            exp_avg_sq.mul_(beta_two).addcmul_(gradient, gradient, value=1.0 - beta_two)
            if config.weight_decay:
                parameter.mul_(1.0 - config.lr * config.weight_decay)
            bias_correction_one = 1.0 - torch.pow(beta_one, step)
            bias_correction_two = 1.0 - torch.pow(beta_two, step)
            denominator = (exp_avg_sq.sqrt() / bias_correction_two.sqrt()).add_(config.eps)
            parameter.sub_(exp_avg / denominator * (config.lr / bias_correction_one))
