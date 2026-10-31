"""Output LoRA: low-rank logits added before sampling, trained by SGD on weighted cross-entropy batches.

The state is shared by all requests (instance scope) or kept per request key (request scope). Training
batches arrive as `pllm.host.WeightedCEBatch` inputs; pllm runs the SGD steps and publishes the new factors.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import torch
from torch.nn import functional as F

import pllm
from pllm import authoring as a


@dataclass(frozen=True, slots=True)
class OutputLoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    learning_rate: float = 1e-3
    steps: int = 1
    seed: int = 0
    scope: pllm.state.StateScope = pllm.state.StateScope.INSTANCE
    key_fields: tuple[pllm.state.RequestKeyField, ...] = (
        pllm.state.RequestKeyField.PROMPT_IDX,
        pllm.state.RequestKeyField.SAMPLE_IDX,
    )

    def __post_init__(self) -> None:
        if (
            min(self.rank, self.steps) < 1
            or not isfinite(self.alpha)
            or self.alpha <= 0
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not 0 <= self.seed < 2**63
            or self.scope not in (pllm.state.StateScope.INSTANCE, pllm.state.StateScope.REQUEST)
        ):
            raise ValueError(
                "output LoRA requires positive rank/steps/scale/rate, 0 <= seed < 2**63, and instance or request scope"
            )


@dataclass(frozen=True)
class OutputLoRAWorkload(pllm.Workload):
    name = "lora.output_head"
    config: OutputLoRAConfig = OutputLoRAConfig()

    @property
    def output_factors(self) -> pllm.InstanceState:
        rank, seed = self.config.rank, self.config.seed

        def initialize(parameters: pllm.state.Tensors, generator: torch.Generator) -> None:
            parameters["a"].normal_(std=0.01, generator=generator.manual_seed(seed))
            parameters["b"].zero_()

        fields: dict[str, Callable[[a.Dimensions], tuple[int, ...]]] = {
            "a": lambda m: (m.hidden_size, rank),
            "b": lambda m: (rank, m.vocab_size),
        }
        if self.config.scope is pllm.state.StateScope.REQUEST:
            return pllm.RequestState(
                fields,
                optimizer=pllm.SGD(self.config.learning_rate),
                initialize=initialize,
                namespace="canonical.output_lora@1",
                keys=self.config.key_fields,
            )
        return pllm.InstanceState(
            fields,
            optimizer=pllm.SGD(self.config.learning_rate),
            initialize=initialize,
            namespace="canonical.output_lora@1",
        )

    def reset(self, parameters: pllm.state.Tensors, generator: torch.Generator) -> None:
        down = parameters["a"]
        down.copy_(
            torch.randn(
                down.shape, device=down.device, dtype=down.dtype, generator=generator.manual_seed(self.config.seed)
            )
            * 0.01
        )
        parameters["b"].zero_()

    @pllm.add_logits()
    def predict(
        self,
        hidden: torch.Tensor = pllm.hidden(-1, pllm.model.HiddenSite.MODEL_OUTPUT),
        down: torch.Tensor = pllm.parameter("output_factors", "a"),
        up: torch.Tensor = pllm.parameter("output_factors", "b"),
    ) -> torch.Tensor:
        return torch.matmul(torch.matmul(hidden.unsqueeze(1), down), up).squeeze(1) * (
            self.config.alpha / self.config.rank
        )

    @pllm.train(
        state="output_factors",
        steps=pllm.Config("config.steps"),
        reset="reset",
    )
    def train(
        self,
        hidden: torch.Tensor = pllm.batch("hidden"),
        logits: torch.Tensor = pllm.batch("base_logits"),
        labels: torch.Tensor = pllm.batch("labels"),
        weights: torch.Tensor = pllm.batch("weights"),
        down: torch.Tensor = pllm.parameter("output_factors", "a"),
        up: torch.Tensor = pllm.parameter("output_factors", "b"),
    ) -> torch.Tensor:
        scores = logits + hidden @ down @ up * (self.config.alpha / self.config.rank)
        row_losses = F.cross_entropy(scores.flatten(0, 1), labels.flatten(), reduction="none").view_as(weights)
        return (row_losses * weights).sum(1) / weights.sum(1).clamp_min(1e-8)


__all__ = ["OutputLoRAConfig", "OutputLoRAWorkload"]
