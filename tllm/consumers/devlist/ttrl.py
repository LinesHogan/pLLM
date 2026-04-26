#!/usr/bin/env python3
"""TTRL-inspired reward grouping and lightweight update consumer."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tllm.consumers.base import BaseConsumer
from tllm.consumers.devlist.common import LowRankAdapter
from tllm.ports.base import ConsumerFlow


@dataclass(frozen=True)
class TTRLSample:
    prompt_idx: int
    sample_idx: int
    completion: str


@dataclass
class TTRLConsumerConfig:
    consumer_id: str = "ttrl"
    enabled: bool = True
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3


@dataclass(frozen=True)
class TTRLStats:
    update_count: int = 0
    loss_count: int = 0
    reward_count: int = 0
    last_loss: float = 0.0


def _normalize_completion(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def majority_vote_rewards(samples: Sequence[TTRLSample]) -> list[float]:
    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        grouped[int(sample.prompt_idx)].append((idx, _normalize_completion(sample.completion)))
    rewards = [0.0] * len(samples)
    for group in grouped.values():
        counts = Counter(completion for _, completion in group)
        best_count = max(counts.values()) if counts else 0
        winner = ""
        for _, completion in group:
            if counts[completion] == best_count:
                winner = completion
                break
        for idx, completion in group:
            rewards[idx] = 1.0 if completion == winner else 0.0
    return rewards


class TTRLEngine(nn.Module):
    def __init__(self, config: TTRLConsumerConfig) -> None:
        super().__init__()
        self.config = config
        self.adapter: LowRankAdapter | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self._update_count = 0
        self._loss_count = 0
        self._reward_count = 0
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
            raise RuntimeError("TTRL update expects 2D hidden rows")
        hidden_size = int(hidden.shape[1])
        if self.adapter is None:
            self._build_adapter(hidden_size)
        elif int(self.config.hidden_size) != hidden_size:
            raise RuntimeError("TTRL hidden size changed after adapter initialization")
        assert self.adapter is not None
        first_param = next(self.adapter.parameters(), None)
        if first_param is not None and (first_param.device != hidden.device or first_param.dtype != hidden.dtype):
            self.to(device=hidden.device, dtype=hidden.dtype)

    def update(self, hidden: torch.Tensor, rewards: torch.Tensor) -> None:
        if not bool(self.config.enabled):
            return
        if int(hidden.shape[0]) != int(rewards.reshape(-1).shape[0]):
            raise RuntimeError("TTRL update requires one reward per hidden row")
        if int(hidden.shape[0]) <= 0:
            return
        self.ensure_shape(hidden)
        assert self.adapter is not None
        assert self.optimizer is not None
        hidden_train = self._detached_train_input(hidden)
        rewards_train = self._detached_train_input(rewards.reshape(-1, 1).to(device=hidden.device, dtype=hidden.dtype))
        pred = self.adapter(hidden_train)
        loss = F.mse_loss(pred, rewards_train)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._update_count += 1
        self._loss_count += 1
        self._reward_count += int(rewards_train.shape[0])
        self._last_loss = float(loss.detach().cpu())

    def read_stats(self) -> TTRLStats:
        return TTRLStats(
            update_count=int(self._update_count),
            loss_count=int(self._loss_count),
            reward_count=int(self._reward_count),
            last_loss=float(self._last_loss),
        )


class TTRLConsumer(BaseConsumer):
    """Post-generation TTRL-style consumer with local majority rewards."""

    def __init__(self, config: TTRLConsumerConfig, *, engine: TTRLEngine | None = None) -> None:
        self.config = config
        self.engine = engine or TTRLEngine(config)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        return ()

    def rewards_for_samples(self, samples: Sequence[TTRLSample]) -> torch.Tensor:
        rewards = majority_vote_rewards(samples)
        return torch.tensor(rewards, dtype=torch.float32)

    def update_from_rewards(self, hidden: torch.Tensor, rewards: torch.Tensor | Iterable[float]) -> None:
        reward_tensor = rewards if isinstance(rewards, torch.Tensor) else torch.tensor(list(rewards), dtype=torch.float32)
        self.engine.update(hidden, reward_tensor)

    def read_stats(self) -> TTRLStats:
        return self.engine.read_stats()
