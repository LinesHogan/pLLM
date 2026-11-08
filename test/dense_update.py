"""Causal chunk mathematics over request-owned, per-target buffers."""

from dataclasses import dataclass
from typing import Literal

import torch

import pllm
from pllm.model import Projection


@dataclass(frozen=True, slots=True)
class DenseUpdateConfig:
    """Chunk mathematics and state visibility for a request-owned dense residual."""

    targets: tuple[Projection, ...]
    chunk_size: int = 4096
    learning_rate: float = 1e-3
    tokens: Literal["prefill", "decode", "all"] = "prefill"  # whose projection operands train the state
    visibility: Literal["forward_pinned", "exact"] = "forward_pinned"

    def __post_init__(self) -> None:
        if (
            not self.targets
            or tuple(sorted(set(self.targets))) != self.targets
            or self.chunk_size < 1
            or self.learning_rate <= 0.0
        ):
            raise ValueError("DenseUpdateConfig needs sorted unique targets, chunk_size >= 1, and learning_rate > 0")


@dataclass(frozen=True)
class DenseUpdate(pllm.Workload):
    """Train dense residuals at the declared projections using carried input/target chunks."""

    config: DenseUpdateConfig
    dtype: torch.dtype = torch.float32

    @property
    def dense_fast_weights(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "delta": pllm.Field(lambda in_features, out_features: (out_features, in_features), dtype=self.dtype),
            },
            key_fields=("request_id", "prompt_index", "sample_index"),
        )

    @pllm.train_in_forward(
        visibility=lambda self: self.config.visibility,
        state="dense_fast_weights",
        targets=lambda self: self.config.targets,
        weight="delta",
        chunk=lambda self: self.config.chunk_size,
        tokens=lambda self: self.config.tokens,
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


__all__ = ["DenseUpdateConfig", "DenseUpdate"]
