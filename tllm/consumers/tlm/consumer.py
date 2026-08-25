#!/usr/bin/env python3
"""TLM ordering around an HF/PEFT q_proj/v_proj side trainer."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

import torch

from tllm.consumers.base import BaseConsumer
from tllm.consumers.tlm.config import TLMConsumerConfig
from tllm.consumers.tlm.trainer import (
    HFPEFTTLMTrainer,
    TLMTrainBatch,
    TLMTrainResult,
)
from tllm.ports.base import ConsumerFlow
from tllm.runtime.test_time_lora import LoRAVersionStore


_T = TypeVar("_T")


class TLMTrainer(Protocol):
    def train_batch(self, batch: TLMTrainBatch, *, mode: str) -> TLMTrainResult: ...


@dataclass(frozen=True, slots=True)
class TLMStats:
    update_count: int = 0
    loss_count: int = 0
    selected_count: int = 0
    buffered_count: int = 0
    last_loss: float = 0.0
    adapter_version: int = 0


class TLMConsumer(BaseConsumer[TLMConsumerConfig]):
    """Official-family TLM: full-model forward, q/v LoRA-only backward."""

    def __init__(
        self,
        config: TLMConsumerConfig,
        *,
        trainer: TLMTrainer | None = None,
        version_store: LoRAVersionStore | None = None,
    ) -> None:
        self.config = config
        self._trainer = trainer
        self._versions = version_store or LoRAVersionStore(config.consumer_id)
        self._offline: list[TLMTrainBatch] = []
        self._update_count = 0
        self._loss_count = 0
        self._selected_count = 0
        self._last_loss = 0.0

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    @property
    def trainer(self) -> TLMTrainer | None:
        return self._trainer

    @property
    def version_store(self) -> LoRAVersionStore:
        return self._versions

    def runtime_adapter_store(self) -> LoRAVersionStore:
        return self._versions

    def flows(self) -> Sequence[ConsumerFlow]:
        # Training consumes tokenized examples, not hidden-state surrogates
        # captured from vLLM inference kernels.
        return ()

    def bind_hf_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self._trainer is not None:
            raise RuntimeError("TLM trainer is already bound")
        trainer = HFPEFTTLMTrainer(model, self.config)
        self._trainer = trainer
        return trainer.model

    def _train(self, batch: TLMTrainBatch) -> None:
        if not self.config.enabled:
            return
        if self._trainer is None:
            raise RuntimeError("TLM requires bind_hf_model() or an injected trainer")
        result = self._trainer.train_batch(batch, mode=self.config.mode)
        self._versions.publish(result.target_modules)
        self._update_count += 1
        self._loss_count += 1
        self._selected_count += int(result.selected_count)
        self._last_loss = float(result.loss)

    def update_online(self, batch: TLMTrainBatch) -> None:
        self._train(batch)

    def update_offline(self, batch: TLMTrainBatch) -> None:
        if self.config.enabled:
            self._offline.append(batch)

    def flush_offline(self) -> None:
        batches = tuple(self._offline)
        self._offline.clear()
        for batch in batches:
            self._train(batch)

    def run_ordered_step(
        self,
        *,
        predict: Callable[[], _T],
        batch: TLMTrainBatch,
    ) -> _T:
        """Online predicts then trains; offline trains then predicts."""

        if self.config.mode == "online":
            result = predict()
            self.update_online(batch)
            return result
        self.update_offline(batch)
        self.flush_offline()
        return predict()

    def read_stats(self) -> TLMStats:
        return TLMStats(
            update_count=self._update_count,
            loss_count=self._loss_count,
            selected_count=self._selected_count,
            buffered_count=sum(int(batch.input_ids.shape[0]) for batch in self._offline),
            last_loss=self._last_loss,
            adapter_version=self._versions.latest_version,
        )


__all__ = ["TLMConsumer", "TLMStats", "TLMTrainer"]
