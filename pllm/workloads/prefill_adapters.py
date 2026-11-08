"""Prefill adapters: each request learns low-rank MLP adapters chunk by chunk while its prompt is read.

The adapter of a layer is trained to reconstruct that layer's MLP input from the down projection; later
chunks of the same prompt already use the update from earlier chunks (visibility="exact"), and decode uses
the adapter the prompt ended with.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

import pllm

TARGETS = tuple(pllm.model.Projection("down_proj", layer) for layer in (0, 2, 4, 6, 8))


def initialize(fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
    """Start with a zero model delta and a nonzero path for the first B gradient."""
    fields["a"].normal_(std=0.02, generator=generator)
    fields["b"].zero_()


@dataclass(frozen=True)
class PrefillAdaptersWorkload(pllm.Workload):
    """A request learns during prefill and uses its frozen final adapter for decode."""

    chunk_size: int = 128
    rank: int = 8
    learning_rate: float = 1e-3
    dtype: torch.dtype | Literal["auto"] = "auto"

    @property
    def adapters(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda in_features, out_features: (self.rank, in_features), dtype=self.dtype),
                "b": pllm.Field(lambda in_features, out_features: (out_features, self.rank), dtype=self.dtype),
            },
            init=pllm.Computed(initialize),
            # Apply the mean reconstruction gradient once per prompt chunk.
            optimizer=pllm.SGD(self.learning_rate),
        )

    def chunk(self) -> int:
        return self.chunk_size

    @pllm.train_in_forward(
        state="adapters",
        targets=TARGETS,
        operator="low_rank",
        visibility="exact",
        chunk=chunk,
        flush="prompt_end",
    )
    def learn(
        self,
        x: torch.Tensor = pllm.operand("input"),
        target: torch.Tensor = pllm.operand("target"),
        base: torch.Tensor = pllm.operand("weight"),
        a: torch.Tensor = pllm.parameter("adapters", "a"),
        b: torch.Tensor = pllm.parameter("adapters", "b"),
    ) -> torch.Tensor:
        """Return each row's reconstruction error with FP32 (or wider) error math."""
        prediction = x @ base.T + (x @ a.to(x.dtype).T) @ b.to(x.dtype).T
        dtype = torch.promote_types(prediction.dtype, torch.float32)
        return (prediction.to(dtype) - target.to(dtype)).square().mean(-1)
