"""Input reconstruction with per-request, gated delta-rule head weights."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

import pllm


@dataclass(frozen=True, slots=True)
class GatedDeltaConfig:
    """Chunk-level gated delta rule with constant, untrained gates."""

    targets: tuple[pllm.model.ParameterTarget, ...]
    head_width: int = 64
    chunk_size: int = 64
    alpha: float = 0.99
    beta: float = 0.1
    update_decode: bool = False
    visibility: pllm.Visibility = pllm.Exact()

    def __post_init__(self) -> None:
        if (
            not self.targets
            or tuple(sorted(set(self.targets))) != self.targets
            or self.head_width < 1
            or self.chunk_size < 1
            or not 0 < self.alpha <= 1
            or not 0 < self.beta < 2
        ):
            raise ValueError(
                "Gated delta needs unique targets, positive head/chunk sizes, 0 < alpha <= 1 and 0 < beta < 2"
            )


@dataclass(frozen=True)
class GatedDeltaWorkload(pllm.Workload):
    """Learn per-request head residuals with exclusive chunk reads by default."""

    config: GatedDeltaConfig

    @property
    def fast_weights(self) -> pllm.RequestState:
        """Declare fp32 head weights for each request."""
        return pllm.RequestState(
            {
                "weights": pllm.Buffer(
                    lambda inputs, outputs: (
                        inputs // self.config.head_width,
                        self.config.head_width,
                        self.config.head_width,
                    ),
                    dtype=torch.float32,
                ),
            },
            namespace="gated_delta.fast_weight.state@1",
        )

    @pllm.overlay(
        visibility=pllm.Config("config.visibility"),
        state="fast_weights",
        targets=pllm.Config("config.targets"),
        weight="weights",
        chunk=pllm.Config("config.chunk_size"),
        operator=pllm.model.ParameterOverlayOperator.HEAD_ADD,
        decode=pllm.Config("config.update_decode"),
    )
    def step(
        self,
        state: torch.Tensor = pllm.parameter("fast_weights", "weights"),
        hidden: torch.Tensor = pllm.operand("input"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One gated MSE update; read each chunk using its incoming state."""
        inputs = hidden.float().reshape(-1, *state.shape[:2])
        keys = F.normalize(inputs, dim=-1)
        values, keys = inputs.transpose(0, 1), keys.transpose(0, 1)
        decayed = state * self.config.alpha
        error = torch.bmm(keys, decayed.transpose(1, 2)) - values
        # Averaging over the chunk's rows keeps alpha*(I - beta*mean(k kᵀ)) contractive for 0 < beta < 2.
        updated = decayed - self.config.beta / self.config.chunk_size * torch.bmm(error.transpose(1, 2), keys)
        output = torch.bmm(values, state.transpose(1, 2)).transpose(0, 1)
        return updated, output.reshape(-1, hidden.shape[1])
