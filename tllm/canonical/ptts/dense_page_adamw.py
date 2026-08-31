"""Typed AdamW layout and CUDA COW kernel for dense-page state."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from tllm.contracts.tensor_layout import LayoutFieldId


@dataclass(frozen=True, slots=True)
class DensePageAdamWFieldBinding:
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
class DensePageAdamWConfig:
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


def apply_cuda_dense_page_adamw(
    *,
    pools: tuple[torch.Tensor, ...],
    source_page_ids: tuple[int, ...],
    target_page_ids: tuple[int, ...],
    fields: DensePageAdamWFieldBinding,
    config: DensePageAdamWConfig,
    gradients: tuple[torch.Tensor, ...],
) -> None:
    """Launch one eager CUDA COW AdamW update into private target rows."""

    parameter_pool = pools[int(fields.parameter)]
    gradient_pool = pools[int(fields.gradient)]
    step_pool = pools[int(fields.step)]
    exp_avg_pool = pools[int(fields.exp_avg)]
    exp_avg_sq_pool = pools[int(fields.exp_avg_sq)]
    beta_one, beta_two = config.betas
    with torch.no_grad():
        for source_page_id, target_page_id, gradient in zip(
            source_page_ids,
            target_page_ids,
            gradients,
            strict=True,
        ):
            for pool in pools:
                pool[target_page_id].copy_(pool[source_page_id])
            gradient_pool[target_page_id].copy_(gradient)
            parameter = parameter_pool[target_page_id]
            step = step_pool[target_page_id]
            exp_avg = exp_avg_pool[target_page_id]
            exp_avg_sq = exp_avg_sq_pool[target_page_id]
            step.add_(1)
            exp_avg.mul_(beta_one).add_(gradient, alpha=1.0 - beta_one)
            exp_avg_sq.mul_(beta_two).addcmul_(
                gradient,
                gradient,
                value=1.0 - beta_two,
            )
            if config.weight_decay:
                parameter.mul_(1.0 - config.lr * config.weight_decay)
            bias_correction_one = 1.0 - torch.pow(beta_one, step)
            bias_correction_two = 1.0 - torch.pow(beta_two, step)
            denominator = (
                exp_avg_sq.sqrt() / bias_correction_two.sqrt()
            ).add_(config.eps)
            parameter.sub_(
                exp_avg / denominator * (config.lr / bias_correction_one)
            )


__all__ = [
    "DensePageAdamWConfig",
    "DensePageAdamWFieldBinding",
    "apply_cuda_dense_page_adamw",
]
