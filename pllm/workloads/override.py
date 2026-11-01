"""Iterative logit heads written as tensor methods, state and sampler effects."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite

import torch
from torch.nn import functional as F

import pllm


@dataclass(frozen=True, slots=True)
class OverRIDEConfig:
    rank: int = 16
    num_iterations: int = 10
    lambd: float = 0.8
    learning_rate: float = 1e-3
    enabled: bool = True

    def __post_init__(self) -> None:
        if (
            min(self.rank, self.num_iterations) < 1
            or not isfinite(self.lambd)
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("OverRIDE requires positive dimensions/rate and finite reweighting")


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """One loss per row; pllm averages the eligible rows."""
    return F.cross_entropy(logits.float(), labels.long(), reduction="none")


def head_logits(hidden: torch.Tensor, down: torch.Tensor, up: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    down, up = down.index_select(0, indices), up.index_select(0, indices)
    return torch.bmm(torch.bmm(hidden.unsqueeze(1), down.transpose(1, 2)), up.transpose(1, 2)).squeeze(1)


HIDDEN = pllm.final_hidden()
SAMPLES = pllm.sample_index(pllm.Attr("config.num_iterations"))
DOWN = pllm.parameter("iteration_heads", "down")
UP = pllm.parameter("iteration_heads", "up")


@dataclass(frozen=True, slots=True)
class OverRIDEWorkload(pllm.Workload):
    config: OverRIDEConfig = OverRIDEConfig()
    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = cross_entropy

    @property
    def iteration_heads(self) -> pllm.InstanceState:
        config = self.config

        def initialize(parameters: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
            parameters["down"].normal_(std=0.02, generator=generator)

        return pllm.InstanceState(
            {
                "down": lambda m: (config.num_iterations - 1, config.rank, m.hidden_size),
                "up": lambda m: (config.num_iterations - 1, m.vocab_size, config.rank),
            },
            optimizer=pllm.AdamW(lr=config.learning_rate),
            initialize=initialize,
        )

    @pllm.add_logits()
    def reweight(
        self,
        hidden: torch.Tensor = HIDDEN,
        samples: torch.Tensor = SAMPLES,
        down: torch.Tensor = DOWN,
        up: torch.Tensor = UP,
    ) -> torch.Tensor:
        if not self.config.enabled or self.config.num_iterations == 1:
            return hidden.new_zeros((len(hidden), up.shape[1]))
        return head_logits(hidden, down, up, (samples - 1).clamp_min(0)) * (
            (samples > 0).unsqueeze(1) * self.config.lambd
        )

    def eligible(self, samples: torch.Tensor = SAMPLES) -> torch.Tensor:
        return (samples < self.config.num_iterations - 1) & self.config.enabled

    @pllm.train(state="iteration_heads", rows="eligible")
    def train(
        self,
        hidden: torch.Tensor = HIDDEN,
        samples: torch.Tensor = SAMPLES,
        down: torch.Tensor = DOWN,
        up: torch.Tensor = UP,
        base: torch.Tensor = pllm.base_logits(),
        tokens: torch.Tensor = pllm.sampled_tokens(),
    ) -> torch.Tensor:
        # Every row evaluates and pllm masks the ineligible ones, so the last sample reads a valid head.
        if self.config.num_iterations == 1:
            return base.new_zeros(len(base))
        heads = samples.clamp_max(self.config.num_iterations - 2)
        return self.objective(base + head_logits(hidden, down, up, heads), tokens)
