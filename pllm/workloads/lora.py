"""Output LoRA: low-rank logits added before sampling, trained by SGD on weighted cross-entropy batches.

The state is shared by all requests (instance scope) or kept per request key (request scope). Training
batches arrive as `pllm.host.WeightedCEBatch` inputs; pllm runs the SGD steps and publishes the new factors.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch
from torch.nn import functional as F

import pllm


@dataclass(frozen=True, slots=True)
class OutputLoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    learning_rate: float = 1e-3
    steps: int = 1
    seed: int = 0
    scope: Literal["instance", "request"] = "instance"
    # The request-state key in request scope; see pllm.RequestState(key_fields=).
    key_fields: tuple[Literal["request_id", "prompt_index", "sample_index", "route"], ...] = (
        "prompt_index",
        "sample_index",
    )

    def __post_init__(self) -> None:
        if (
            min(self.rank, self.steps) < 1
            or not isfinite(self.alpha)
            or self.alpha <= 0
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or not 0 <= self.seed < 2**63
            or self.scope not in ("instance", "request")
        ):
            raise ValueError(
                "output LoRA requires positive rank/steps/scale/rate, 0 <= seed < 2**63, and instance or request scope"
            )


@dataclass(frozen=True)
class OutputLoRAWorkload(pllm.Workload):
    config: OutputLoRAConfig = OutputLoRAConfig()

    @property
    def output_factors(self) -> pllm.InstanceState:
        rank, seed = self.config.rank, self.config.seed

        def initialize(parameters: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
            parameters["a"].normal_(std=0.01, generator=generator.manual_seed(seed))
            parameters["b"].zero_()

        fields: dict[str, Callable[[pllm.Dimensions], tuple[int, ...]]] = {
            "a": lambda m: (m.hidden_size, rank),
            "b": lambda m: (rank, m.vocab_size),
        }
        if self.config.scope == "request":
            return pllm.RequestState(
                fields,
                optimizer=pllm.SGD(self.config.learning_rate),
                initialize=initialize,
                key_fields=self.config.key_fields,
            )
        return pllm.InstanceState(
            fields,
            optimizer=pllm.SGD(self.config.learning_rate),
            initialize=initialize,
        )

    def reset(self, parameters: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
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
        hidden: torch.Tensor = pllm.final_hidden(),
        down: torch.Tensor = pllm.parameter("output_factors", "a"),
        up: torch.Tensor = pllm.parameter("output_factors", "b"),
    ) -> torch.Tensor:
        return torch.matmul(torch.matmul(hidden.unsqueeze(1), down), up).squeeze(1) * (
            self.config.alpha / self.config.rank
        )

    @pllm.train(
        state="output_factors",
        steps=pllm.Attr("config.steps"),
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
