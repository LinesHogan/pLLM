#!/usr/bin/env python3
"""TLM-inspired online/offline TTL consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tllm.consumers.base import BaseConsumer
from tllm.consumers.devlist.common import LowRankAdapter
from tllm.ports.base import ConsumerFlow


TLMMode = Literal["online", "offline"]


@dataclass
class TLMConsumerConfig:
    consumer_id: str = "tlm"
    enabled: bool = True
    mode: TLMMode = "online"
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3
    threshold: float = 0.5
    adapter_slots: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("online", "offline"):
            raise ValueError("TLMConsumerConfig mode must be 'online' or 'offline'")
        self.adapter_slots = max(1, int(self.adapter_slots))


@dataclass(frozen=True)
class TLMStats:
    update_count: int = 0
    loss_count: int = 0
    selected_count: int = 0
    buffered_count: int = 0
    last_loss: float = 0.0


def select_ttl_samples(scores: torch.Tensor, *, threshold: float) -> list[int]:
    flat = scores.reshape(-1).detach().cpu()
    return [idx for idx, score in enumerate(flat.tolist()) if float(score) < float(threshold)]


class TLMEngine(nn.Module):
    def __init__(self, config: TLMConsumerConfig) -> None:
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleList()
        self.optimizer: torch.optim.Optimizer | None = None
        self._buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._update_count = 0
        self._loss_count = 0
        self._selected_count = 0
        self._last_loss = 0.0
        if int(config.hidden_size) > 0:
            self._build_adapters(int(config.hidden_size))

    @staticmethod
    def _detached_train_input(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach()
        if torch.is_inference(out):
            out = out.clone()
        return out

    def _build_adapters(self, hidden_size: int) -> None:
        self.config.hidden_size = int(hidden_size)
        self.adapters = nn.ModuleList(
            [
                LowRankAdapter(in_dim=int(hidden_size), out_dim=1, rank=int(self.config.rank))
                for _ in range(int(self.config.adapter_slots))
            ]
        )
        self.optimizer = torch.optim.Adam(self.adapters.parameters(), lr=float(self.config.learning_rate))

    def ensure_shape(self, hidden: torch.Tensor) -> None:
        if hidden.ndim != 2:
            raise RuntimeError("TLM update expects 2D hidden rows")
        hidden_size = int(hidden.shape[1])
        if not self.adapters:
            self._build_adapters(hidden_size)
        elif int(self.config.hidden_size) != hidden_size:
            raise RuntimeError("TLM hidden size changed after adapter initialization")
        first_param = next(self.adapters.parameters(), None)
        if first_param is not None and (first_param.device != hidden.device or first_param.dtype != hidden.dtype):
            self.to(device=hidden.device, dtype=hidden.dtype)

    def _update_selected(self, hidden: torch.Tensor, scores: torch.Tensor) -> None:
        if not bool(self.config.enabled):
            return
        selected = select_ttl_samples(scores, threshold=float(self.config.threshold))
        if not selected:
            return
        self.ensure_shape(hidden)
        assert self.optimizer is not None
        row_ids = torch.as_tensor(selected, device=hidden.device, dtype=torch.long)
        hidden_train = self._detached_train_input(hidden).index_select(0, row_ids)
        score_targets = self._detached_train_input(scores.reshape(-1).to(device=hidden.device, dtype=hidden.dtype)).index_select(0, row_ids)
        desired = (1.0 - score_targets).reshape(-1, 1)
        pred = self.adapters[0](hidden_train)
        loss = F.mse_loss(pred, desired)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._selected_count += len(selected)
        self._update_count += 1
        self._loss_count += 1
        self._last_loss = float(loss.detach().cpu())

    def update_online(self, hidden: torch.Tensor, scores: torch.Tensor) -> None:
        self._update_selected(hidden, scores)

    def update_offline(self, hidden: torch.Tensor, scores: torch.Tensor) -> None:
        if not bool(self.config.enabled):
            return
        self._buffer.append((self._detached_train_input(hidden), self._detached_train_input(scores)))

    def flush_offline(self) -> None:
        if not self._buffer:
            return
        hidden = torch.cat([item[0] for item in self._buffer], dim=0)
        scores = torch.cat([item[1].reshape(-1) for item in self._buffer], dim=0)
        self._buffer.clear()
        self._update_selected(hidden, scores)

    def read_stats(self) -> TLMStats:
        return TLMStats(
            update_count=int(self._update_count),
            loss_count=int(self._loss_count),
            selected_count=int(self._selected_count),
            buffered_count=sum(int(item[0].shape[0]) for item in self._buffer),
            last_loss=float(self._last_loss),
        )


class TLMConsumer(BaseConsumer):
    """Local online/offline test-time learning consumer."""

    def __init__(self, config: TLMConsumerConfig, *, engine: TLMEngine | None = None) -> None:
        self.config = config
        self.engine = engine or TLMEngine(config)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        return ()

    def update_online(self, hidden: torch.Tensor, scores: torch.Tensor) -> None:
        self.engine.update_online(hidden, scores)

    def update_offline(self, hidden: torch.Tensor, scores: torch.Tensor) -> None:
        self.engine.update_offline(hidden, scores)

    def flush_offline(self) -> None:
        self.engine.flush_offline()

    def read_stats(self) -> TLMStats:
        return self.engine.read_stats()
