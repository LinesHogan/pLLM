"""Vanilla seed edited by the algorithm-search author; scored in a separate process."""

from dataclasses import dataclass

import torch

import pllm


@dataclass(frozen=True)
class Vanilla(pllm.Workload):
    """An explicit ordinary residual sum, preserving vanilla model generation."""

    @pllm.custom_residual(site="mlp", layers=(0,))
    def residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Keep the model's ordinary residual addition."""
        return x + y


WORKLOAD = Vanilla()
NEUTRAL = Vanilla()
