#!/usr/bin/env python3
"""FTTT-inspired feedback edit consumer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tllm.consumers.base import BaseConsumer
from tllm.consumers.devlist.common import LowRankAdapter
from tllm.ports.base import ConsumerFlow


@dataclass(frozen=True)
class FeedbackRecord:
    prompt_idx: int
    feedback: str
    reward: float


@dataclass
class FTTTConsumerConfig:
    consumer_id: str = "fttt"
    enabled: bool = True
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3
    max_queue_size: int = 128


@dataclass(frozen=True)
class FTTTStats:
    queued_feedback: int = 0
    dropped_feedback: int = 0
    feedback_count: int = 0
    update_count: int = 0
    loss_count: int = 0
    last_loss: float = 0.0


class FTTTEngine(nn.Module):
    def __init__(self, config: FTTTConsumerConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter: LowRankAdapter | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self._feedback_count = 0
        self._update_count = 0
        self._loss_count = 0
        self._last_loss = 0.0
        if int(config.hidden_size) > 0:
            self._build_adapter(int(config.hidden_size))

    @staticmethod
    def _detached_train_input(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach()
        if torch.is_inference(out):
            out = out.clone()
        return out

    def _build_adapter(self, hidden_size: int) -> None:
        self.config.hidden_size = int(hidden_size)
        self.adapter = LowRankAdapter(in_dim=int(hidden_size), out_dim=1, rank=int(self.config.rank))
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=float(self.config.learning_rate))

    def ensure_shape(self, hidden: torch.Tensor) -> None:
        if hidden.ndim != 2:
            raise RuntimeError("FTTT update expects 2D hidden rows")
        hidden_size = int(hidden.shape[1])
        if self.adapter is None:
            self._build_adapter(hidden_size)
        elif int(self.config.hidden_size) != hidden_size:
            raise RuntimeError("FTTT hidden size changed after adapter initialization")
        assert self.adapter is not None
        first_param = next(self.adapter.parameters(), None)
        if first_param is not None and (first_param.device != hidden.device or first_param.dtype != hidden.dtype):
            self.to(device=hidden.device, dtype=hidden.dtype)

    def update(self, hidden: torch.Tensor, feedback: Sequence[FeedbackRecord]) -> None:
        if not bool(self.config.enabled):
            return
        if int(hidden.shape[0]) != len(feedback):
            raise RuntimeError("FTTT update requires one feedback record per hidden row")
        if len(feedback) <= 0:
            return
        self.ensure_shape(hidden)
        assert self.adapter is not None
        assert self.optimizer is not None
        hidden_train = self._detached_train_input(hidden)
        rewards = torch.tensor(
            [float(record.reward) for record in feedback],
            device=hidden.device,
            dtype=hidden.dtype,
        ).reshape(-1, 1)
        pred = self.adapter(hidden_train)
        loss = F.mse_loss(pred, rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._feedback_count += len(feedback)
        self._update_count += 1
        self._loss_count += 1
        self._last_loss = float(loss.detach().cpu())

    def read_stats(self, *, queued_feedback: int, dropped_feedback: int) -> FTTTStats:
        return FTTTStats(
            queued_feedback=int(queued_feedback),
            dropped_feedback=int(dropped_feedback),
            feedback_count=int(self._feedback_count),
            update_count=int(self._update_count),
            loss_count=int(self._loss_count),
            last_loss=float(self._last_loss),
        )


class FTTTConsumer(BaseConsumer):
    """Feedback-conditioned local edit consumer."""

    def __init__(self, config: FTTTConsumerConfig, *, engine: FTTTEngine | None = None) -> None:
        self.config = config
        self.engine = engine or FTTTEngine(config)
        self._queue: deque[FeedbackRecord] = deque()
        self._dropped_feedback = 0

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        return ()

    def enqueue_feedback(self, feedback: FeedbackRecord) -> bool:
        if len(self._queue) >= max(0, int(self.config.max_queue_size)):
            self._dropped_feedback += 1
            return False
        self._queue.append(feedback)
        return True

    def drain_feedback(self, *, limit: int = 0) -> list[FeedbackRecord]:
        max_items = len(self._queue) if int(limit) <= 0 else min(int(limit), len(self._queue))
        return [self._queue.popleft() for _ in range(max_items)]

    def apply_feedback_update(self, hidden: torch.Tensor, feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord]) -> None:
        records = list(feedback)
        self.engine.update(hidden, records)

    def read_stats(self) -> FTTTStats:
        return self.engine.read_stats(queued_feedback=len(self._queue), dropped_feedback=self._dropped_feedback)
