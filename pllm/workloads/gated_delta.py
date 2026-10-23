"""Input reconstruction with per-request, gated delta-rule head weights."""

from dataclasses import dataclass
from typing import Annotated

import torch
from torch.nn import functional as F

import pllm
from pllm.kernels.head_overlay import head_correction
from pllm.workloads.chunk_tail import join_tail, save_tail

Input = Annotated[torch.Tensor, pllm.Operand.INPUT]
Weight = Annotated[torch.Tensor, pllm.Operand.WEIGHT]
State = Annotated[torch.Tensor, pllm.Parameter("fast_weights", "weights")]
Tail = Annotated[torch.Tensor, pllm.Parameter("fast_weights", "tail")]
Length = Annotated[torch.Tensor, pllm.Parameter("fast_weights", "length")]


@dataclass(frozen=True, slots=True)
class GatedDeltaConfig:
    """Chunk-level gated delta rule with constant, untrained gates."""

    targets: tuple[pllm.ParameterTarget, ...]
    head_width: int = 64
    chunk_size: int = 64
    alpha: float = 0.99
    beta: float = 0.1
    update_decode: bool = False

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
    """Learn a chunk-level gated delta-rule residual after each model forward.

    Queries read the state pinned at forward start. Training returns exclusive
    chunk reads; visibility of those reads inside the model arrives with δ4.
    """

    config: GatedDeltaConfig

    @property
    def fast_weights(self) -> pllm.RequestState:
        """Declare fp32 head weights and the unfinished input chunk for each request."""
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
                "tail": pllm.Buffer(lambda inputs, outputs: (max(1, self.config.chunk_size - 1), inputs)),
                "length": pllm.Buffer((), dtype=torch.int64),
            },
            namespace="gated_delta.fast_weight.state@1",
        )

    def step(self, state: torch.Tensor, inputs: torch.Tensor, keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One gated MSE update; read each chunk using its incoming state."""
        values, keys = inputs.transpose(0, 1), keys.transpose(0, 1)
        decayed = state * self.config.alpha
        error = torch.bmm(keys, decayed.transpose(1, 2)) - values
        # Averaging over the chunk's rows keeps alpha*(I - beta*mean(k kᵀ)) contractive for 0 < beta < 2.
        updated = decayed - self.config.beta / self.config.chunk_size * torch.bmm(error.transpose(1, 2), keys)
        output = torch.bmm(values, state.transpose(1, 2)).transpose(0, 1)
        return updated, output

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(
        visibility=pllm.Visibility.forward_pinned(),
        state="fast_weights",
        targets=pllm.Config("config.targets"),
        weight="weights",
        operator=pllm.ParameterOverlayOperator.HEAD_ADD,
        decode=pllm.Config("config.update_decode"),
    )
    def transition(self, hidden: Input, base: Weight, weights: State, tail: Tail, length: Length) -> pllm.TensorUpdate:
        """Train complete chunks, retain the tail and return the training readouts."""
        chunk = self.config.chunk_size
        (rows,) = join_tail((tail,), (hidden,), length, chunk)
        trained = len(rows) // chunk * chunk
        inputs = rows.float().reshape(-1, *weights.shape[:2])
        keys = F.normalize(inputs, dim=-1)
        updated, outputs = pllm.scan(self.step, carry=weights, over=(inputs[:trained], keys[:trained]), chunk=chunk)
        weights.copy_(updated)
        remaining = save_tail((tail,), (rows,), length, trained)
        corrections = tuple(output.reshape(-1, hidden.shape[1]) for output in outputs)
        if remaining:
            corrections += (head_correction(rows[trained:], weights),)
        output = hidden @ base.T + torch.cat(corrections)[-len(hidden) :].to(hidden.dtype)
        return pllm.TensorUpdate(output, trained > 0, trained // chunk, trained, {"tail_rows": remaining})
