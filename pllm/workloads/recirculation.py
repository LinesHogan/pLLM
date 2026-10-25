"""Paper-style serial recirculation, expressed as a user layer route."""

from dataclasses import dataclass

import torch

import pllm


def norm_match(
    source: torch.Tensor, destination: torch.Tensor, alpha: float | torch.Tensor, beta: float | torch.Tensor
) -> torch.Tensor:
    """Match source norm in fp32, clamping only its norm, then mix residual streams."""
    src, dst = source.float(), destination.float()
    matched = src * (
        torch.ops.aten.linalg_vector_norm.default(dst, 2, [-1], True)
        / torch.ops.aten.linalg_vector_norm.default(src, 2, [-1], True).clamp_min(torch.finfo(torch.float32).tiny)
    )
    return (beta * dst + alpha * matched).to(destination.dtype)


@dataclass(frozen=True)
class RecirculationWorkload(pllm.Workload):
    """Overwrite upper-layer KV after each normal token, keeping normal-pass logits."""

    name = "recirculation"
    source: int = -1
    destination: int = 0
    alpha: float = 0.15
    beta: float | None = None
    ramp_tokens: int = 0

    @pllm.layer_route(tokens="all")
    def recirculate(self, layers: pllm.Layers) -> torch.Tensor:
        """Run the normal stack, then replay above the mixed destination tap."""
        taps = layers()
        a = (
            self.alpha * (layers.position.float().clamp(max=self.ramp_tokens) / self.ramp_tokens)
            if self.ramp_tokens
            else self.alpha
        )
        b = 1 - a if self.beta is None else self.beta
        mixed = norm_match(taps[self.source], taps[self.destination], a, b)
        destination = self.destination if self.destination >= 0 else layers.layer_count + self.destination
        layers(start=destination + 1, hidden=mixed)
        return taps[-1]
