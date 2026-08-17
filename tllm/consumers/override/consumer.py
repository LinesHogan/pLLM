#!/usr/bin/env python3
"""BaseConsumer wrapper around the OverRIDE reweighting engine."""

from __future__ import annotations

from typing import Sequence

import torch

from tllm.consumers.base import BaseConsumer
from tllm.consumers.override.config import OverRIDEConsumerConfig
from tllm.consumers.override.engine import (
    OverRIDEBatchLogitsResult,
    OverRIDELogitsResult,
    OverRIDEReweightEngine,
    OverRIDEStats,
)
from tllm.ports.base import ConsumerFlow

__all__ = [
    "OverRIDEBatchLogitsResult",
    "OverRIDEConsumer",
    "OverRIDEConsumerConfig",
    "OverRIDELogitsResult",
    "OverRIDEReweightEngine",
    "OverRIDEStats",
]


class OverRIDEConsumer(BaseConsumer[OverRIDEConsumerConfig]):
    """tLLM-native consumer shell for OverRIDE-style iterative reweighting."""

    def __init__(self, config: OverRIDEConsumerConfig, *, engine: OverRIDEReweightEngine | None = None) -> None:
        self.config = config
        self.engine = engine or OverRIDEReweightEngine(config)
        from tllm.consumers.override.sampler_provider import OverRIDESamplerModifierProvider

        self._sampler_provider = OverRIDESamplerModifierProvider(engine=self.engine)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        return ()

    def sampler_modifier_provider(self):
        return self._sampler_provider

    def reweight_logits(self, hidden: torch.Tensor, logits: torch.Tensor, *, iteration: int) -> OverRIDELogitsResult:
        return self.engine.reweight_logits(hidden, logits, iteration=iteration)

    def iterations_from_request_ids(self, request_ids: Sequence[str]) -> tuple[int, ...]:
        return self.engine.iterations_from_request_ids(request_ids)

    def reweight_logits_batch(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        *,
        request_ids: Sequence[str] | None = None,
        iterations: Sequence[int] | None = None,
        row_ids: tuple[int, ...] | torch.Tensor | None = None,
    ) -> OverRIDEBatchLogitsResult:
        return self.engine.reweight_logits_batch(
            hidden,
            logits,
            request_ids=request_ids,
            iterations=iterations,
            row_ids=row_ids,
        )

    def update_from_sampled_tokens(self, q_logits_next: torch.Tensor | None, sampled_token_ids: torch.Tensor) -> None:
        self.engine.update_from_sampled_tokens(q_logits_next, sampled_token_ids)

    def update_from_sampled_tokens_batch(
        self,
        sampled_token_ids: torch.Tensor,
        *,
        row_ids: tuple[int, ...] | torch.Tensor | None = None,
    ) -> None:
        self.engine.update_from_sampled_tokens_batch(sampled_token_ids, row_ids=row_ids)

    def read_stats(self) -> OverRIDEStats:
        return self.engine.read_stats()
