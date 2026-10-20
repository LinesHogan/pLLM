"""Local down-projection reconstruction learned once per prefill chunk."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

import torch

import pllm

_TARGET = pllm.ParameterTarget(0, pllm.ParameterModuleKind.MLP, pllm.ParameterRole.DOWN_PROJECTION_WEIGHT)


def initialize(fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
    """Start with a zero model delta and a nonzero path for the first B gradient."""
    fields["a"].normal_(std=0.02, generator=generator)
    fields["b"].zero_()


@dataclass(frozen=True)
class PrefillAdapters(pllm.Workload):
    """A request learns during prefill and uses its frozen final adapter for decode."""

    name = "prefill_adapters"
    chunk_size: int = 128
    rank: int = 8
    learning_rate: float = 1e-3
    dtype: torch.dtype = torch.bfloat16

    @property
    def adapters(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Buffer(lambda inputs, outputs: (self.rank, inputs), self.dtype),
                "b": pllm.Buffer(lambda inputs, outputs: (outputs, self.rank), self.dtype),
            },
            namespace="example.prefill_adapters@1",
            dtype=self.dtype,
            initialize=initialize,
            optimizer=pllm.SGD(self.learning_rate),
        )

    @property
    def prefill(self) -> pllm.ModelScan:
        return pllm.ModelScan(
            self.chunk_size,
            pllm.ParameterEffect("adapters", (_TARGET,), operator=pllm.ParameterOverlayOperator.LOW_RANK_ADD),
            learn="loss",
            algorithm=pllm.Exact(),
            kv=pllm.KVPolicy.CAUSAL_HISTORY,
        )

    @pllm.train(state="adapters")
    def loss(
        self,
        x: Annotated[torch.Tensor, pllm.Operand.INPUT],
        target: Annotated[torch.Tensor, pllm.Operand.TARGET],
        base: Annotated[torch.Tensor, pllm.Operand.WEIGHT],
        a: Annotated[torch.Tensor, pllm.Parameter("adapters", "a")],
        b: Annotated[torch.Tensor, pllm.Parameter("adapters", "b")],
    ) -> torch.Tensor:
        """Reconstruct this layer's MLP input without differentiating the base model."""
        prediction = x @ base.T + (x @ a.T) @ b.T
        dtype = torch.promote_types(prediction.dtype, torch.float32)
        error = prediction.to(dtype) - target.to(dtype)
        return error.square().mean()
