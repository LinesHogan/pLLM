"""Exact and Jacobi-prefill recirculation tutorials built from public layer routes."""

from dataclasses import dataclass

import torch

import pllm


def norm_match(
    source: torch.Tensor, destination: torch.Tensor, alpha: float | torch.Tensor, beta: float | torch.Tensor
) -> torch.Tensor:
    """Match source norm in fp32, clamping only its norm, then mix residual streams."""
    src, dst = source.float(), destination.float()
    matched = src * (
        dst.square().sum(-1, keepdim=True).sqrt()
        / src.square().sum(-1, keepdim=True).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
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

    def destination_layer(self, layers: pllm.Layers) -> int:
        """Resolve the destination against the model's layer count."""
        return self.destination if self.destination >= 0 else layers.layer_count + self.destination

    def mix(self, taps: pllm.Taps, destination: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Mix source and destination streams with the position-dependent ramp."""
        a = (
            self.alpha * (position.float().clamp(max=self.ramp_tokens) / self.ramp_tokens)
            if self.ramp_tokens
            else self.alpha
        )
        b = 1 - a if self.beta is None else self.beta
        return norm_match(taps[self.source], destination, a, b)

    def step(self, layers: pllm.Layers) -> torch.Tensor:
        """Run the normal stack, then replay above the mixed destination tap."""
        taps = layers()
        layers(start=self.destination_layer(layers) + 1, hidden=self.mix(taps, taps[self.destination], layers.position))
        return taps[-1]

    @pllm.layer_route(tokens="all")
    def recirculate(self, layers: pllm.Layers) -> torch.Tensor:
        """Apply exact recirculation to every token."""
        return self.step(layers)


@dataclass(frozen=True)
class JacobiRecirculation(RecirculationWorkload):
    """Recirculation whose prompt is solved by rounds Jacobi sweeps instead of token by token."""

    rounds: int = 2

    def __post_init__(self) -> None:
        if type(self.rounds) is not int or self.rounds < 1:
            raise ValueError("set rounds to at least 1 for Jacobi prefill")

    @pllm.layer_route(tokens="decode")
    def recirculate(self, layers: pllm.Layers) -> torch.Tensor:
        """Keep exact per-token recirculation during decoding."""
        return self.step(layers)

    @pllm.layer_route(tokens="prefill", over="span")
    def prefill(self, layers: pllm.Layers) -> torch.Tensor:
        """Commit a seed pass, then read prior sweeps without overwriting their KV."""
        d = self.destination_layer(layers)
        taps = layers()
        destination = taps[d]
        for round in range(self.rounds):
            if round:
                taps = layers(start=d + 1, hidden=destination, kv="peek")
            layers(start=d + 1, hidden=self.mix(taps, destination, layers.position))
        return taps[-1]
