"""ESamp: each request trains a small distiller that predicts a deep hidden state from a shallow one,
and the prediction steers sampling once the distiller has trained."""

from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F

import pllm


@dataclass(frozen=True, slots=True)
class ESampConfig:
    rank: int = 128
    source_layer: int = 0
    target_layer: int = -1
    output_layernorm: bool = True
    intervention_beta: float = 0.0  # 0 trains the distiller without guiding sampling
    learning_rate: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.rank < 1 or not isfinite(self.intervention_beta) or self.intervention_beta < 0.0:
            raise ValueError("ESamp needs rank >= 1 and a finite, nonnegative intervention_beta")


class Distiller(nn.Module):
    """A gated low-rank residual on the source hidden state, optionally layer-normed."""

    def __init__(self, width: int, rank: int, normalize: bool) -> None:
        super().__init__()
        self.up = nn.Parameter(torch.empty(width, rank).normal_(0.0, 0.02))
        self.gate = nn.Parameter(torch.empty(width, rank).normal_(0.0, 0.02))
        self.down = nn.Parameter(torch.empty(rank, width).normal_(0.0, 0.02))
        self.bias = nn.Parameter(torch.ones(rank))
        self.scale = nn.Parameter(torch.ones(width if normalize else 0))
        self.shift = nn.Parameter(torch.zeros(width if normalize else 0))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        output = source + (source @ self.up * F.silu(source @ self.gate + self.bias)) @ self.down
        if not len(self.scale):
            return output
        return F.layer_norm(output, (source.shape[-1],)) * self.scale + self.shift


@dataclass(frozen=True)
class ESampWorkload(pllm.Workload):
    config: ESampConfig = ESampConfig()

    @property
    def distiller(self) -> pllm.RequestState:
        config = self.config
        return pllm.RequestState(
            module=lambda m: Distiller(m.hidden_size, config.rank, config.output_layernorm),
            optimizer=pllm.AdamW(config.learning_rate, config.betas, config.eps, config.weight_decay),
        )

    def trained(self, versions: torch.Tensor = pllm.version("distiller")) -> torch.Tensor:
        return versions > 0

    def beta(self) -> float:
        return self.config.intervention_beta

    def source_layer(self) -> int:
        return self.config.source_layer

    def target_layer(self) -> int:
        return self.config.target_layer

    @pllm.guide_sampling(beta=beta, rows="trained")
    def predict(
        self,
        source: torch.Tensor = pllm.hidden(source_layer),
        distiller: Distiller = pllm.module("distiller"),
    ) -> torch.Tensor:
        return distiller(source)

    @pllm.train(state="distiller")
    def loss(
        self,
        source: torch.Tensor = pllm.hidden(source_layer),
        target: torch.Tensor = pllm.hidden(target_layer),
        distiller: Distiller = pllm.module("distiller"),
    ) -> torch.Tensor:
        return (distiller(source) - target).square().mean(1)


__all__ = ["ESampConfig", "ESampWorkload"]
