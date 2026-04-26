#!/usr/bin/env python3
"""OverRIDE-inspired iterative reweighting consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tllm.consumers.base import BaseConsumer
from tllm.consumers.devlist.common import LowRankAdapter
from tllm.ports.base import ConsumerFlow


@dataclass
class OverRIDEConsumerConfig:
    consumer_id: str = "override"
    enabled: bool = True
    hidden_size: int = 0
    vocab_size: int = 0
    rank: int = 16
    num_iterations: int = 2
    lambd: float = 0.8
    learning_rate: float = 1e-4


@dataclass(frozen=True)
class OverRIDELogitsResult:
    logits: torch.Tensor
    q_logits_next: torch.Tensor | None


@dataclass(frozen=True)
class OverRIDEStats:
    reweight_count: int = 0
    modified_row_count: int = 0
    loss_count: int = 0
    update_count: int = 0
    last_loss: float = 0.0


class OverRIDEReweightEngine(nn.Module):
    """Local low-rank reweighting engine based on the OverRIDE core formula."""

    def __init__(self, config: OverRIDEConsumerConfig) -> None:
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList()
        self.optimizer: torch.optim.Optimizer | None = None
        self._reweight_count = 0
        self._modified_row_count = 0
        self._loss_count = 0
        self._update_count = 0
        self._last_loss = 0.0
        if self._has_shape:
            self._build_heads()

    @property
    def _has_shape(self) -> bool:
        return int(self.config.hidden_size) > 0 and int(self.config.vocab_size) > 0

    def _build_heads(self) -> None:
        num_heads = max(0, int(self.config.num_iterations) - 1)
        self.heads = nn.ModuleList(
            [
                LowRankAdapter(
                    in_dim=int(self.config.hidden_size),
                    out_dim=int(self.config.vocab_size),
                    rank=int(self.config.rank),
                )
                for _ in range(num_heads)
            ]
        )
        self.optimizer = torch.optim.Adam(self.heads.parameters(), lr=float(self.config.learning_rate))

    def ensure_shape(self, *, hidden_size: int, vocab_size: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._has_shape and self.heads:
            first_param = next(self.heads.parameters(), None)
            if first_param is not None and (first_param.device != device or first_param.dtype != dtype):
                self.to(device=device, dtype=dtype)
            return
        self.config.hidden_size = int(hidden_size)
        self.config.vocab_size = int(vocab_size)
        self._build_heads()
        self.to(device=device, dtype=dtype)

    def _head_for_iteration(self, iteration: int) -> LowRankAdapter | None:
        idx = int(iteration) - 1
        if idx < 0 or idx >= len(self.heads):
            return None
        return self.heads[idx]

    @staticmethod
    def _detached_train_input(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach()
        if torch.is_inference(out):
            out = out.clone()
        return out

    def reweight_logits(self, hidden: torch.Tensor, logits: torch.Tensor, *, iteration: int) -> OverRIDELogitsResult:
        if not bool(self.config.enabled):
            return OverRIDELogitsResult(logits=logits, q_logits_next=None)
        if hidden.ndim != 2 or logits.ndim != 2:
            raise RuntimeError("OverRIDE reweighting expects 2D hidden and logits tensors")
        if int(hidden.shape[0]) != int(logits.shape[0]):
            raise RuntimeError("OverRIDE reweighting requires hidden/logits row counts to match")
        self.ensure_shape(
            hidden_size=int(hidden.shape[1]),
            vocab_size=int(logits.shape[1]),
            device=hidden.device,
            dtype=hidden.dtype,
        )

        previous_head = self._head_for_iteration(int(iteration) - 1)
        current_head = self._head_for_iteration(int(iteration))
        reweighted = logits
        hidden_for_head = self._detached_train_input(hidden)
        logits_for_update = self._detached_train_input(logits)
        if previous_head is not None:
            reweighted = logits - float(self.config.lambd) * previous_head(hidden_for_head).to(dtype=logits.dtype)
            self._reweight_count += 1
            self._modified_row_count += int(logits.shape[0])

        q_logits_next = None
        if current_head is not None:
            q_logits_next = logits_for_update + current_head(hidden_for_head).to(dtype=logits.dtype)
        return OverRIDELogitsResult(logits=reweighted, q_logits_next=q_logits_next)

    def update_from_sampled_tokens(self, q_logits_next: torch.Tensor | None, sampled_token_ids: torch.Tensor) -> None:
        if q_logits_next is None or not bool(self.config.enabled):
            return
        if self.optimizer is None:
            raise RuntimeError("OverRIDE update requires initialized reweighting heads")
        targets = sampled_token_ids.reshape(-1).to(device=q_logits_next.device, dtype=torch.long)
        q_logits = q_logits_next.reshape(-1, int(q_logits_next.shape[-1]))
        if int(targets.numel()) != int(q_logits.shape[0]):
            raise RuntimeError("OverRIDE update requires one sampled token per q-logits row")
        loss = F.cross_entropy(q_logits, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._loss_count += 1
        self._update_count += 1
        self._last_loss = float(loss.detach().cpu())

    def read_stats(self) -> OverRIDEStats:
        return OverRIDEStats(
            reweight_count=int(self._reweight_count),
            modified_row_count=int(self._modified_row_count),
            loss_count=int(self._loss_count),
            update_count=int(self._update_count),
            last_loss=float(self._last_loss),
        )


class OverRIDEConsumer(BaseConsumer):
    """tLLM-native consumer shell for OverRIDE-style iterative reweighting."""

    def __init__(self, config: OverRIDEConsumerConfig, *, engine: OverRIDEReweightEngine | None = None) -> None:
        self.config = config
        self.engine = engine or OverRIDEReweightEngine(config)

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        return ()

    def reweight_logits(self, hidden: torch.Tensor, logits: torch.Tensor, *, iteration: int) -> OverRIDELogitsResult:
        return self.engine.reweight_logits(hidden, logits, iteration=iteration)

    def update_from_sampled_tokens(self, q_logits_next: torch.Tensor | None, sampled_token_ids: torch.Tensor) -> None:
        self.engine.update_from_sampled_tokens(q_logits_next, sampled_token_ids)

    def read_stats(self) -> OverRIDEStats:
        return self.engine.read_stats()
