"""Causal chunk mathematics over request-owned, per-target buffers."""

from dataclasses import dataclass
from typing import Annotated

import torch

import pllm
from pllm import ConditionalLayerReplay, ParameterTarget, TensorUpdate
from pllm.workloads.chunk_tail import join_tail, save_tail

Input = Annotated[torch.Tensor, pllm.Operand.INPUT]
Target = Annotated[torch.Tensor, pllm.Operand.TARGET]
Weight = Annotated[torch.Tensor, pllm.Operand.WEIGHT]
Delta = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "delta")]
TailInput = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_input")]
TailTarget = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_target")]
TailLength = Annotated[torch.Tensor, pllm.Parameter("dense_fast_weights", "tail_length")]


@dataclass(frozen=True, slots=True)
class InPlaceTTTConfig:
    targets: tuple[ParameterTarget, ...]
    chunk_size: int = 4096
    learning_rate: float = 1e-3
    update_prefill: bool = True
    update_decode: bool = False
    replay: ConditionalLayerReplay | None = None

    def __post_init__(self) -> None:
        if (
            not self.targets
            or tuple(sorted(set(self.targets))) != self.targets
            or self.chunk_size < 1
            or self.learning_rate <= 0.0
        ):
            raise ValueError("In-Place-TTT mathematical configuration is invalid")


@dataclass(frozen=True)
class InPlaceTTTWorkload(pllm.Workload):
    name = "inplace_ttt.dense_fast_weight"
    config: InPlaceTTTConfig

    @property
    def dense_fast_weights(self) -> pllm.RequestState:
        tail = max(1, self.config.chunk_size - 1)
        return pllm.RequestState(
            {
                "delta": pllm.Buffer(lambda inputs, outputs: (outputs, inputs)),
                "tail_input": pllm.Buffer(lambda inputs, outputs: (tail, inputs)),
                "tail_target": pllm.Buffer(lambda inputs, outputs: (tail, outputs)),
                "tail_length": pllm.Buffer((), dtype=torch.int64),
            },
            namespace="canonical.dense_fast_weight.state@1",
            keys=(pllm.RequestKeyField.REQUEST_ID, pllm.RequestKeyField.PROMPT_IDX, pllm.RequestKeyField.SAMPLE_IDX),
        )

    def step(
        self, delta: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = torch.mm(inputs, (base + delta).T)
        return delta + (self.config.learning_rate * torch.mm(targets.T, inputs)).to(delta), output

    @pllm.on(pllm.Phase.MODEL_UPDATE)
    @pllm.overlay(
        state="dense_fast_weights",
        targets=pllm.Config("config.targets"),
        weight="delta",
        prefill=pllm.Config("config.update_prefill"),
        decode=pllm.Config("config.update_decode"),
        replay=pllm.Config("config.replay"),
    )
    def transition(
        self,
        hidden: Input,
        target: Target,
        base: Weight,
        delta: Delta,
        tail_input: TailInput,
        tail_target: TailTarget,
        tail_length: TailLength,
    ) -> pllm.TensorUpdate:
        chunk = self.config.chunk_size
        inputs, targets = join_tail((tail_input, tail_target), (hidden, target), tail_length, chunk)
        trained = len(inputs) // chunk * chunk
        updated, outputs = pllm.scan(
            self.step, carry=delta, over=(inputs[:trained], targets[:trained]), chunk=chunk, invariants=(base,)
        )
        delta.copy_(updated)
        remaining = save_tail((tail_input, tail_target), (inputs, targets), tail_length, trained)
        if remaining:
            outputs += (inputs[trained:] @ (base + delta).T,)
        return TensorUpdate(
            torch.cat(outputs)[-len(hidden) :], trained > 0, trained // chunk, trained, {"tail_rows": remaining}
        )


__all__ = ["InPlaceTTTConfig", "InPlaceTTTWorkload"]
