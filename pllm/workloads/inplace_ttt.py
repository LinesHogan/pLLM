"""Causal chunk mathematics over request-owned, per-target buffers."""

from dataclasses import dataclass
from typing import Literal

import torch

import pllm
from pllm.model import ParameterTarget


@dataclass(frozen=True, slots=True)
class InPlaceTTTConfig:
    """Chunk mathematics and state visibility for a request-owned dense residual."""

    targets: tuple[ParameterTarget, ...]
    chunk_size: int = 4096
    learning_rate: float = 1e-3
    tokens: Literal["prefill", "decode", "all"] = "prefill"  # whose projection operands train the state
    visibility: pllm.Visibility = pllm.Visibility.forward_pinned()

    def __post_init__(self) -> None:
        if (
            not self.targets
            or tuple(sorted(set(self.targets))) != self.targets
            or self.chunk_size < 1
            or self.learning_rate <= 0.0
        ):
            raise ValueError("InPlaceTTTConfig needs sorted unique targets, chunk_size >= 1, and learning_rate > 0")


@dataclass(frozen=True)
class InPlaceTTTWorkload(pllm.Workload):
    """Train dense residuals at the declared projections using carried input/target chunks."""

    config: InPlaceTTTConfig
    dtype: torch.dtype = torch.float32

    @property
    def dense_fast_weights(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "delta": pllm.Buffer(lambda inputs, outputs: (outputs, inputs), dtype=self.dtype),
            },
            keys=(
                pllm.state.RequestKeyField.REQUEST_ID,
                pllm.state.RequestKeyField.PROMPT_INDEX,
                pllm.state.RequestKeyField.SAMPLE_INDEX,
            ),
        )

    @pllm.train_in_forward(
        visibility=pllm.Config("config.visibility"),
        state="dense_fast_weights",
        targets=pllm.Config("config.targets"),
        weight="delta",
        chunk=pllm.Config("config.chunk_size"),
        tokens=pllm.Config("config.tokens"),
    )
    def step(
        self,
        delta: torch.Tensor = pllm.parameter("dense_fast_weights", "delta"),
        inputs: torch.Tensor = pllm.operand("input"),
        targets: torch.Tensor = pllm.operand("target"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the incoming residual and advance one additive chunk."""
        output = torch.mm(inputs, delta.to(inputs.dtype).T)
        return delta + (self.config.learning_rate * torch.mm(targets.T, inputs)).to(delta), output


__all__ = ["InPlaceTTTConfig", "InPlaceTTTWorkload"]
