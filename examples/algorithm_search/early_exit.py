"""Bounded early-exit loop seed for algorithm search on Qwen2.5-0.5B."""

from dataclasses import dataclass

import torch

import pllm


@dataclass(frozen=True)
class EarlyExit(pllm.Workload):
    """Skip the last two layers for rows below the hidden-energy threshold."""

    threshold: float = 0.0

    @pllm.custom_forward(tokens="decode")
    def forward(self, layers: pllm.model.Layers) -> torch.Tensor:
        """Visit two tail layers while the row's active mask remains true."""
        h = layers(stop=22)[21]
        active = torch.ones_like(h[:, :1], dtype=torch.bool)
        for layer in range(22, 24):
            active = active & (h.float().square().mean(-1, keepdim=True) > self.threshold)
            h = layers(layer, layer + 1, hidden=h, active=active)[layer]
        return h


WORKLOAD = EarlyExit()
NEUTRAL = EarlyExit(threshold=-1.0)
