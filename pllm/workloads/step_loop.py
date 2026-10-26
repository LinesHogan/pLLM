"""Carry each token's pre-norm residual into layer zero at generated positions."""

from dataclasses import dataclass

import torch

import pllm


@dataclass(frozen=True)
class StepLoop(pllm.Workload):
    """Use prompt embeddings, then recurrent pre-norm hidden states during generation."""

    history = pllm.PositionState()

    @pllm.layer_route(tokens="all")
    def loop(self, layers: pllm.Layers) -> torch.Tensor:
        """Save every token so cached prompt blocks also carry the boundary state."""
        previous = layers.load("history", offset=-1)
        taps = layers(hidden=torch.where(layers.generated, previous, layers.seed))
        layers.save("history", taps[-1])
        return taps[-1]
