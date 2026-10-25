"""Prefill adapters: each request learns low-rank MLP adapters chunk by chunk while its prompt is read.

The adapter of a layer is trained to reconstruct that layer's MLP input from the down projection; later
chunks of the same prompt already use the update from earlier chunks (Exact visibility), and decode uses
the adapter the prompt ended with.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

import pllm

TARGETS = tuple(
    pllm.model.ParameterTarget(
        layer, pllm.model.ParameterModuleKind.MLP, pllm.model.ParameterRole.DOWN_PROJECTION_WEIGHT
    )
    for layer in (0, 2, 4, 6, 8)
)


def initialize(fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
    """Start with a zero model delta and a nonzero path for the first B gradient."""
    fields["a"].normal_(std=0.02, generator=generator)
    fields["b"].zero_()


def reconstruction_loss(
    x: torch.Tensor, target: torch.Tensor, base: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """How far the adapted down projection is from this layer's MLP input."""
    prediction = x @ base.T + (x @ a.T) @ b.T
    dtype = torch.promote_types(prediction.dtype, torch.float32)
    return (prediction.to(dtype) - target.to(dtype)).square().mean()


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
            dtype=self.dtype,
            initialize=initialize,
        )

    @pllm.overlay(
        state="adapters",
        targets=TARGETS,
        operator=pllm.model.ParameterOverlayOperator.LOW_RANK_ADD,
        visibility=pllm.Exact(),
    )
    def learn(
        self,
        x: torch.Tensor = pllm.operand("input"),
        target: torch.Tensor = pllm.operand("target"),
        base: torch.Tensor = pllm.operand("weight"),
        a: torch.Tensor = pllm.parameter("adapters", "a"),
        b: torch.Tensor = pllm.parameter("adapters", "b"),
    ) -> pllm.TensorUpdate:
        """Apply the current adapter to each chunk, then take one SGD step on that chunk."""
        outputs: list[torch.Tensor] = []
        for start in range(0, len(x), self.chunk_size):
            rows, wanted = x[start : start + self.chunk_size], target[start : start + self.chunk_size]
            outputs.append((rows @ a.T) @ b.T)
            with torch.inference_mode(False), torch.enable_grad():
                leaves = a.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
                loss = reconstruction_loss(rows.detach().clone(), wanted.detach().clone(), base.detach(), *leaves)
                grad_a, grad_b = torch.autograd.grad(loss, leaves)
            a.sub_(self.learning_rate * grad_a)
            b.sub_(self.learning_rate * grad_b)
        chunks = torch.full((), len(outputs), device=x.device)
        return pllm.TensorUpdate(torch.cat(outputs), chunks > 0, chunks, torch.full((), len(x), device=x.device))
