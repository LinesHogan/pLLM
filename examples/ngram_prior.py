"""A small hashed n-gram logits prior using token history; fit the table offline for a real application."""

from dataclasses import dataclass

import torch

import pllm


@dataclass(frozen=True)
class NGramPrior(pllm.Workload):
    """Add an offline-fitted bigram prior; scale=0 leaves the model unchanged."""

    path: str
    buckets: int = 256
    scale: float = 0.1

    @property
    def memory(self) -> pllm.InstanceState:
        return pllm.InstanceState({"table": lambda m: (self.buckets, m.vocab_size)}, init=pllm.Pretrained(self.path))

    def lookup(
        self,
        ids: torch.Tensor = pllm.input_tokens(history=1),
        table: torch.Tensor = pllm.parameter("memory", "table"),
    ) -> torch.Tensor:
        keys = (ids[:, 0] * 31 + ids[:, 1] + 1) % self.buckets
        return table[keys] * self.scale

    @pllm.add_logits()
    def prior(self, logits: torch.Tensor = pllm.result("lookup")) -> torch.Tensor:
        return logits
