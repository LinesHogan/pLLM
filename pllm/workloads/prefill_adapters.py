"""Prefill adapters: each request learns low-rank MLP adapters chunk by chunk while its prompt is read.

The adapter of a layer is trained to reconstruct that layer's MLP input from the down projection; later
chunks of the same prompt already use the update from earlier chunks (Exact visibility), and decode uses
the adapter the prompt ended with.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

import pllm
from pllm.workloads.chunk_tail import join_tail, save_tail

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
    x: torch.Tensor, target: torch.Tensor, base: torch.Tensor, a: torch.Tensor, b: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    """How far the adapted down projection is from this layer's MLP input."""
    prediction = x @ base.T + (x @ a.T) @ b.T
    dtype = torch.promote_types(prediction.dtype, torch.float32)
    return (prediction.to(dtype) - target.to(dtype)).square().sum() / (count.clamp_min(1) * target.shape[1])


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
                "tail_input": pllm.Buffer(lambda inputs, outputs: (max(1, self.chunk_size - 1), inputs), self.dtype),
                "tail_target": pllm.Buffer(lambda inputs, outputs: (max(1, self.chunk_size - 1), outputs), self.dtype),
                "tail_length": pllm.Buffer((), torch.int64),
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
        tail_input: torch.Tensor = pllm.parameter("adapters", "tail_input"),
        tail_target: torch.Tensor = pllm.parameter("adapters", "tail_target"),
        tail_length: torch.Tensor = pllm.parameter("adapters", "tail_length"),
        prompt_end: torch.Tensor = pllm.operand("prompt_end"),
    ) -> pllm.TensorUpdate:
        """Read prompt-relative chunks exclusively; flush a short chunk only at prompt end."""
        (inputs, targets), valid = join_tail((tail_input, tail_target), (x, target), tail_length, self.chunk_size)
        positions = torch.arange(len(x), device=x.device) + tail_length
        trained = torch.where(prompt_end, valid, valid // self.chunk_size * self.chunk_size)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(inputs), self.chunk_size):
            rows, wanted = inputs[start : start + self.chunk_size], targets[start : start + self.chunk_size]
            outputs.append((rows @ a.T) @ b.T)
            count = (valid - start).clamp(0, self.chunk_size)
            active = start < trained
            with torch.inference_mode(False), torch.enable_grad():
                leaves = a.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
                loss = reconstruction_loss(
                    rows.detach().clone(), wanted.detach().clone(), base.detach(), *leaves, count
                )
                grad_a, grad_b = torch.autograd.grad(loss, leaves)
            a.sub_(self.learning_rate * torch.where(active, grad_a, 0))
            b.sub_(self.learning_rate * torch.where(active, grad_b, 0))
        remaining = save_tail((tail_input, tail_target), (inputs, targets), tail_length, valid, trained)
        chunks = (trained + self.chunk_size - 1) // self.chunk_size
        return pllm.TensorUpdate(
            torch.cat(outputs).index_select(0, positions), trained > 0, chunks, trained, {"tail_rows": remaining}
        )
