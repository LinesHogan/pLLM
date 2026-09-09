"""Independent request-local low-rank Huber learning with persistent AdamW."""

from collections.abc import Mapping
from typing import Annotated

import torch
import pllm


def initialize(parameters: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
    parameters["a"].normal_(0.0, 0.02, generator=generator)
    parameters["b"].normal_(0.0, 0.02, generator=generator)


class HuberWorkload(pllm.Workload):
    def __init__(self, rank: int = 2) -> None:
        self.rank = rank

    @property
    def adapter(self) -> pllm.RequestState:
        rank = self.rank
        return pllm.RequestState({"a": lambda m: (m.hidden_size, rank),
            "b": lambda m: (rank, m.hidden_size)}, initialize=initialize,
            optimizer=pllm.AdamW(lr=0.03, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.1))

    def ready(self, versions: Annotated[torch.Tensor, pllm.StateVersion("adapter")]) -> torch.Tensor:
        return versions > 0

    def forward(self, source: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        latent = torch.bmm(source.unsqueeze(1), a)
        return source + torch.bmm(latent, b).squeeze(1)

    @pllm.on(pllm.Phase.BEFORE_SAMPLE, rows="ready")
    @pllm.guide_sampling(beta=0.2)
    def predict(self, source: Annotated[torch.Tensor, pllm.Hidden(0)],
                a: Annotated[torch.Tensor, pllm.Parameter("adapter", "a")],
                b: Annotated[torch.Tensor, pllm.Parameter("adapter", "b")],
                versions: Annotated[torch.Tensor, pllm.StateVersion("adapter")]) -> torch.Tensor:
        return self.forward(source, a, b)

    @pllm.on(pllm.Phase.AFTER_SAMPLE)
    @pllm.train(state="adapter", reduction="state_mean")
    def loss(self, source: Annotated[torch.Tensor, pllm.Hidden(0)],
             target: Annotated[torch.Tensor, pllm.Hidden(-1)],
             a: Annotated[torch.Tensor, pllm.Parameter("adapter", "a")],
             b: Annotated[torch.Tensor, pllm.Parameter("adapter", "b")],
             groups: Annotated[torch.Tensor, pllm.StateIndex("adapter")]) -> torch.Tensor:
        losses = torch.nn.functional.huber_loss(self.forward(source, a, b), target,
            delta=0.25, reduction="none").mean(1)
        counts = torch.bincount(groups).to(losses)
        return losses.new_zeros(counts.shape).index_add(0, groups, losses) / counts
