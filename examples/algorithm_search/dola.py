"""DoLa seed for algorithm search, using the built-in layer contrast."""

from dataclasses import dataclass

import torch

import pllm
from pllm.workloads.dola import DoLaWorkload


@dataclass(frozen=True)
class Vanilla(pllm.Workload):
    """Neutral residual sum for this standalone seed."""

    @pllm.custom_residual(site="mlp", layers=(0,))
    def residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Keep the model's ordinary residual addition."""
        return x + y


WORKLOAD = DoLaWorkload(layers=(0, 4, 8, 12))
NEUTRAL = Vanilla()
