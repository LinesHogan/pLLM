"""Local down-projection reconstruction learned once per prefill chunk."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

import pllm

_TARGETS = tuple(
    pllm.model.ParameterTarget(
        layer, pllm.model.ParameterModuleKind.MLP, pllm.model.ParameterRole.DOWN_PROJECTION_WEIGHT
    )
    for layer in (0, 2, 4, 6, 8)
)


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
    def prefill(self) -> pllm.model.ModelScan:
        return pllm.model.ModelScan(
            self.chunk_size,
            pllm.model.ParameterEffect("adapters", _TARGETS, operator=pllm.model.ParameterOverlayOperator.LOW_RANK_ADD),
            learn="loss",
            visibility=pllm.Exact(),
            kv=pllm.model.KVPolicy.CAUSAL_HISTORY,
        )

    @pllm.train(state="adapters")
    def loss(
        self,
        x: torch.Tensor = pllm.operand("input"),
        target: torch.Tensor = pllm.operand("target"),
        base: torch.Tensor = pllm.operand("weight"),
        a: torch.Tensor = pllm.parameter("adapters", "a"),
        b: torch.Tensor = pllm.parameter("adapters", "b"),
    ) -> torch.Tensor:
        """Reconstruct this layer's MLP input without differentiating the base model."""
        prediction = x @ base.T + (x @ a.T) @ b.T
        dtype = torch.promote_types(prediction.dtype, torch.float32)
        error = prediction.to(dtype) - target.to(dtype)
        return error.square().mean()
