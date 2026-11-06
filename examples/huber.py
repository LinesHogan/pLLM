"""Independent request-local low-rank Huber learning with persistent AdamW."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

import pllm


def initialize(parameters: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
    parameters["a"].normal_(0.0, 0.02, generator=generator)
    parameters["b"].normal_(0.0, 0.02, generator=generator)


@dataclass(frozen=True)
class HuberWorkload(pllm.Workload):
    """Best-practice user workload: an importable frozen dataclass that EngineCore receives by pickle."""

    rank: int = 2

    @property
    def adapter(self) -> pllm.RequestState:
        rank = self.rank
        return pllm.RequestState(
            {"a": lambda m: (m.hidden_size, rank), "b": lambda m: (rank, m.hidden_size)},
            initialize=initialize,
            optimizer=pllm.AdamW(lr=0.03, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.1),
        )

    def ready(self, versions: torch.Tensor = pllm.version("adapter")) -> torch.Tensor:
        return versions > 0

    def forward(self, source: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        latent = torch.bmm(source.unsqueeze(1), a)
        return source + torch.bmm(latent, b).squeeze(1)

    @pllm.guide_sampling(beta=0.2, rows="ready")
    def predict(
        self,
        source: torch.Tensor = pllm.hidden(0),
        a: torch.Tensor = pllm.parameter("adapter", "a"),
        b: torch.Tensor = pllm.parameter("adapter", "b"),
    ) -> torch.Tensor:
        return self.forward(source, a, b)

    @pllm.train(state="adapter")
    def loss(
        self,
        source: torch.Tensor = pllm.hidden(0),
        target: torch.Tensor = pllm.hidden(-1),
        a: torch.Tensor = pllm.parameter("adapter", "a"),
        b: torch.Tensor = pllm.parameter("adapter", "b"),
    ) -> torch.Tensor:
        return torch.nn.functional.huber_loss(self.forward(source, a, b), target, delta=0.25, reduction="none").mean(1)
