#!/usr/bin/env python3
"""OverRIDE-inspired iterative reweighting consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

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
class OverRIDEBatchLogitsResult:
    logits: torch.Tensor
    q_logits_next: tuple[torch.Tensor, ...]
    q_indices: tuple[torch.Tensor, ...]
    iterations: tuple[int, ...]


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
        self._pending_q_logits_next: dict[Hashable, tuple[int, torch.Tensor]] = {}
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
    def iteration_from_request_id(request_id: str) -> int:
        if "_" not in str(request_id):
            return 1
        idx_str = str(request_id).split("_", 1)[0]
        try:
            return int(idx_str) + 1
        except ValueError:
            return 1

    @classmethod
    def iterations_from_request_ids(cls, request_ids: Sequence[str]) -> tuple[int, ...]:
        return tuple(cls.iteration_from_request_id(request_id) for request_id in request_ids)

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

    @staticmethod
    def _default_row_ids(row_count: int) -> tuple[int, ...]:
        return tuple(range(int(row_count)))

    @staticmethod
    def _validate_row_ids(row_ids: Sequence[Hashable], row_count: int) -> tuple[Hashable, ...]:
        normalized = tuple(row_ids)
        if len(normalized) != int(row_count):
            raise RuntimeError("OverRIDE runtime bridge requires one row id per logits row")
        return normalized

    def _resolve_iterations(
        self,
        *,
        row_count: int,
        request_ids: Sequence[str] | None,
        iterations: Sequence[int] | None,
    ) -> tuple[int, ...]:
        if request_ids is not None and iterations is not None:
            raise RuntimeError("OverRIDE runtime bridge accepts request_ids or iterations, not both")
        if iterations is None:
            if request_ids is None:
                raise RuntimeError("OverRIDE runtime bridge requires request_ids or explicit iterations")
            resolved = self.iterations_from_request_ids(request_ids)
        else:
            resolved = tuple(int(iteration) for iteration in iterations)
        if len(resolved) != int(row_count):
            raise RuntimeError("OverRIDE runtime bridge requires one iteration id per logits row")
        return resolved

    def reweight_logits_batch(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        *,
        request_ids: Sequence[str] | None = None,
        iterations: Sequence[int] | None = None,
        row_ids: Sequence[Hashable] | None = None,
    ) -> OverRIDEBatchLogitsResult:
        if hidden.ndim != 2 or logits.ndim != 2:
            raise RuntimeError("OverRIDE runtime bridge expects 2D hidden and logits tensors")
        if int(hidden.shape[0]) != int(logits.shape[0]):
            raise RuntimeError("OverRIDE runtime bridge requires hidden/logits row counts to match")

        row_count = int(logits.shape[0])
        resolved_iterations = self._resolve_iterations(
            row_count=row_count,
            request_ids=request_ids,
            iterations=iterations,
        )
        normalized_row_ids = self._validate_row_ids(
            self._default_row_ids(row_count) if row_ids is None else row_ids,
            row_count,
        )
        self._pending_q_logits_next.clear()

        grouped_row_indices: dict[int, list[int]] = {}
        for row_idx, iteration in enumerate(resolved_iterations):
            grouped_row_indices.setdefault(int(iteration), []).append(row_idx)

        reweighted_logits = torch.empty_like(logits)
        q_logits_next: list[torch.Tensor] = []
        q_indices: list[torch.Tensor] = []
        q_iterations: list[int] = []
        for iteration, row_indices in grouped_row_indices.items():
            index = torch.tensor(row_indices, device=logits.device, dtype=torch.long)
            result = self.reweight_logits(hidden.index_select(0, index), logits.index_select(0, index), iteration=iteration)
            reweighted_logits.index_copy_(0, index, result.logits)
            if result.q_logits_next is None:
                continue
            q_logits_next.append(result.q_logits_next)
            q_indices.append(index)
            q_iterations.append(int(iteration))
            for local_idx, row_idx in enumerate(row_indices):
                self._pending_q_logits_next[normalized_row_ids[row_idx]] = (
                    int(iteration),
                    result.q_logits_next[local_idx : local_idx + 1],
                )

        return OverRIDEBatchLogitsResult(
            logits=reweighted_logits,
            q_logits_next=tuple(q_logits_next),
            q_indices=tuple(q_indices),
            iterations=tuple(q_iterations),
        )

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

    def update_from_sampled_tokens_batch(
        self,
        sampled_token_ids: torch.Tensor,
        *,
        row_ids: Sequence[Hashable] | None = None,
    ) -> None:
        if not bool(self.config.enabled) or not self._pending_q_logits_next:
            return
        targets = sampled_token_ids.reshape(-1)
        target_count = int(targets.numel())
        normalized_row_ids = self._validate_row_ids(
            self._default_row_ids(target_count) if row_ids is None else row_ids,
            target_count,
        )
        selected_q_logits: list[torch.Tensor] = []
        selected_targets: list[torch.Tensor] = []
        for token_idx, row_id in enumerate(normalized_row_ids):
            pending = self._pending_q_logits_next.pop(row_id, None)
            if pending is None:
                continue
            _iteration, q_logits = pending
            selected_q_logits.append(q_logits)
            selected_targets.append(targets[token_idx : token_idx + 1])
        if not selected_q_logits:
            return
        self.update_from_sampled_tokens(torch.cat(selected_q_logits, dim=0), torch.cat(selected_targets, dim=0))

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

    def iterations_from_request_ids(self, request_ids: Sequence[str]) -> tuple[int, ...]:
        return self.engine.iterations_from_request_ids(request_ids)

    def reweight_logits_batch(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        *,
        request_ids: Sequence[str] | None = None,
        iterations: Sequence[int] | None = None,
        row_ids: Sequence[Hashable] | None = None,
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
        row_ids: Sequence[Hashable] | None = None,
    ) -> None:
        self.engine.update_from_sampled_tokens_batch(sampled_token_ids, row_ids=row_ids)

    def read_stats(self) -> OverRIDEStats:
        return self.engine.read_stats()
