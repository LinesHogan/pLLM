#!/usr/bin/env python3
"""TLM-inspired online/offline TTL consumer."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Callable, Literal, Sequence, TypeVar

import torch
from torch import nn
from torch.nn import functional as F

from tllm.contracts.test_time_lora import LoRAScope, TestTimeLoRAUpdate, TestTimeLoRAWeights
from tllm.consumers.base import BaseConsumer
from tllm.consumers._common.alignment import (
    LowRankAdapter,
    detached_train_input,
    select_rows_above_threshold,
    sentence_cross_entropy,
    weighted_one_hot_kl_loss,
)
from tllm.ports.base import ConsumerFlow


TLMMode = Literal["online", "offline"]
_T = TypeVar("_T")


def _require_lora_export_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    return value


def _require_lora_export_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be real")
    return float(value)


def _require_lora_export_request_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError("request_ids must be tuple")
    return value


@dataclass
class TLMConsumerConfig:
    consumer_id: str = "tlm"
    enabled: bool = True
    mode: TLMMode = "online"
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3
    threshold: float = 0.5
    lamb: float = 1.0
    adapter_slots: int = 1
    vocab_size: int = 0
    ignore_index: int = -100

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


@dataclass(frozen=True)
class _TLMLogitBatch:
    hidden: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    selection_logits: torch.Tensor


def select_ttl_samples(sentence_ce: torch.Tensor, *, threshold: float) -> list[int]:
    """Select raw sentence-CE rows following upstream TTL: CE > threshold."""
    return select_rows_above_threshold(sentence_ce, threshold=float(threshold))


def select_scalar_score_fallback_samples(scores: torch.Tensor, *, threshold: float) -> list[int]:
    """Legacy score fallback: lower scalar scores are treated as harder rows."""
    flat = scores.reshape(-1).detach().cpu()
    return [idx for idx, score in enumerate(flat.tolist()) if float(score) < float(threshold)]


def run_tlm_ordered_step(mode: TLMMode, *, predict: Callable[[], _T], train: Callable[[], object]) -> _T:
    """Run the observable upstream TTL ordering for a single step."""
    if mode == "online":
        result = predict()
        train()
        return result
    if mode == "offline":
        train()
        return predict()
    raise ValueError("TLM ordered step mode must be 'online' or 'offline'")


class TLMEngine(nn.Module):
    def __init__(self, config: TLMConsumerConfig) -> None:
        super().__init__()
        self.config = config
        self.adapters = nn.ModuleList()
        self.optimizer: torch.optim.Optimizer | None = None
        self._logit_buffer: list[_TLMLogitBatch] = []
        self._score_fallback_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._update_count = 0
        self._loss_count = 0
        self._selected_count = 0
        self._last_loss = 0.0
        if int(config.hidden_size) > 0:
            self._build_adapters(int(config.hidden_size), max(1, int(config.vocab_size)))

    @staticmethod
    def _detached_train_input(tensor: torch.Tensor) -> torch.Tensor:
        return detached_train_input(tensor)

    def _build_adapters(self, hidden_size: int, out_dim: int) -> None:
        self.config.hidden_size = int(hidden_size)
        self.config.vocab_size = int(out_dim)
        self.adapters = nn.ModuleList(
            [
                LowRankAdapter(in_dim=int(hidden_size), out_dim=int(out_dim), rank=int(self.config.rank))
                for _ in range(int(self.config.adapter_slots))
            ]
        )
        self.optimizer = torch.optim.Adam(self.adapters.parameters(), lr=float(self.config.learning_rate))

    def ensure_shape(self, hidden: torch.Tensor, *, out_dim: int = 1) -> None:
        if hidden.ndim != 2:
            raise RuntimeError("TLM update expects 2D hidden rows")
        hidden_size = int(hidden.shape[1])
        if not self.adapters:
            self._build_adapters(hidden_size, int(out_dim))
        elif int(self.config.hidden_size) != hidden_size:
            raise RuntimeError("TLM hidden size changed after adapter initialization")
        elif int(self.adapters[0].out_dim) != int(out_dim):
            if self._update_count > 0 or self._loss_count > 0:
                raise RuntimeError("TLM adapter output size changed after training started")
            self._build_adapters(hidden_size, int(out_dim))
        first_param = next(self.adapters.parameters(), None)
        if first_param is not None and (first_param.device != hidden.device or first_param.dtype != hidden.dtype):
            self.to(device=hidden.device, dtype=hidden.dtype)

    def _adapter_logits(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        adapter_delta = self.adapters[0](hidden)
        if logits.ndim == 2:
            return logits + adapter_delta
        if logits.ndim == 3:
            return logits + adapter_delta.unsqueeze(1).expand(-1, int(logits.shape[1]), -1)
        raise RuntimeError("TLM logits update expects 2D or 3D logits")

    def _update_logits(self, hidden: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, selection_logits: torch.Tensor) -> None:
        if not bool(self.config.enabled):
            return
        if int(hidden.shape[0]) != int(logits.shape[0]) or int(hidden.shape[0]) != int(labels.reshape(labels.shape[0], -1).shape[0]):
            raise RuntimeError("TLM hidden, logits, and labels must have the same row count")
        vocab_size = int(logits.shape[-1])
        self.ensure_shape(hidden, out_dim=vocab_size)
        assert self.optimizer is not None
        hidden_train = self._detached_train_input(hidden)
        logits_train = self._detached_train_input(logits).to(device=hidden.device, dtype=hidden.dtype)
        labels_train = self._detached_train_input(labels).to(device=hidden.device)
        selection_logits_train = self._detached_train_input(selection_logits).to(device=hidden.device, dtype=hidden.dtype)
        sentence_ce = sentence_cross_entropy(
            selection_logits_train,
            labels_train,
            ignore_index=int(self.config.ignore_index),
        )
        selected = select_ttl_samples(sentence_ce, threshold=float(self.config.threshold))
        if not selected:
            return
        adapted_logits = self._adapter_logits(hidden_train, logits_train)
        loss = weighted_one_hot_kl_loss(
            adapted_logits,
            labels_train,
            sentence_ce=sentence_ce.to(device=adapted_logits.device, dtype=adapted_logits.dtype),
            threshold=float(self.config.threshold),
            lamb=float(self.config.lamb),
            ignore_index=int(self.config.ignore_index),
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._selected_count += len(selected)
        self._update_count += 1
        self._loss_count += 1
        self._last_loss = float(loss.detach().cpu())

    def _update_selected_scores(self, hidden: torch.Tensor, scores: torch.Tensor) -> None:
        if not bool(self.config.enabled):
            return
        selected = select_scalar_score_fallback_samples(scores, threshold=float(self.config.threshold))
        if not selected:
            return
        self.ensure_shape(hidden, out_dim=1)
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

    def update_online(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor | None = None,
        *,
        logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        allow_scalar_fallback: bool = False,
    ) -> None:
        if not bool(self.config.enabled):
            return
        if logits is not None or labels is not None:
            if logits is None or labels is None:
                raise RuntimeError("TLM logits update requires both logits and labels")
            self._update_logits(hidden, logits, labels, logits if base_logits is None else base_logits)
            return
        if scores is not None:
            self._update_selected_scores(hidden, scores)
            return
        raise RuntimeError("TLM online update requires logits/labels or explicit scalar fallback scores")

    def update_offline(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor | None = None,
        *,
        logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        allow_scalar_fallback: bool = False,
    ) -> None:
        if not bool(self.config.enabled):
            return
        if logits is not None or labels is not None:
            if logits is None or labels is None:
                raise RuntimeError("TLM offline logits update requires both logits and labels")
            self._logit_buffer.append(
                _TLMLogitBatch(
                    hidden=self._detached_train_input(hidden),
                    logits=self._detached_train_input(logits),
                    labels=self._detached_train_input(labels),
                    selection_logits=self._detached_train_input(logits if base_logits is None else base_logits),
                )
            )
            return
        if scores is not None:
            self._score_fallback_buffer.append((self._detached_train_input(hidden), self._detached_train_input(scores)))
            return
        raise RuntimeError("TLM offline update requires logits/labels or explicit scalar fallback scores")

    def flush_offline(self) -> None:
        if self._logit_buffer:
            hidden = torch.cat([item.hidden for item in self._logit_buffer], dim=0)
            logits = torch.cat([item.logits for item in self._logit_buffer], dim=0)
            labels = torch.cat([item.labels.reshape(item.labels.shape[0], -1) for item in self._logit_buffer], dim=0)
            if all(item.labels.ndim == 1 for item in self._logit_buffer):
                labels = labels.reshape(-1)
            selection_logits = torch.cat([item.selection_logits for item in self._logit_buffer], dim=0)
            self._logit_buffer.clear()
            self._update_logits(hidden, logits, labels, selection_logits)
        if not self._score_fallback_buffer:
            return
        hidden = torch.cat([item[0] for item in self._score_fallback_buffer], dim=0)
        scores = torch.cat([item[1].reshape(-1) for item in self._score_fallback_buffer], dim=0)
        self._score_fallback_buffer.clear()
        self._update_selected_scores(hidden, scores)

    def read_stats(self) -> TLMStats:
        return TLMStats(
            update_count=int(self._update_count),
            loss_count=int(self._loss_count),
            selected_count=int(self._selected_count),
            buffered_count=sum(int(item.hidden.shape[0]) for item in self._logit_buffer)
            + sum(int(item[0].shape[0]) for item in self._score_fallback_buffer),
            last_loss=float(self._last_loss),
        )

    def export_test_time_lora_update(
        self,
        *,
        adapter_key: str,
        lora_int_id: int,
        target_module_name: str,
        rank: int,
        scope: LoRAScope = "global",
        request_ids: tuple[str, ...] = (),
        step_id: int = -1,
        adapter_slot: int = 0,
        alpha: float = 1.0,
    ) -> TestTimeLoRAUpdate:
        slot = _require_lora_export_int(adapter_slot, field_name="adapter_slot")
        update_rank = _require_lora_export_int(rank, field_name="rank")
        _require_lora_export_int(lora_int_id, field_name="lora_int_id")
        _require_lora_export_int(step_id, field_name="step_id")
        export_request_ids = _require_lora_export_request_ids(request_ids)
        export_alpha = _require_lora_export_real(alpha, field_name="alpha")
        if slot < 0 or slot >= len(self.adapters):
            raise RuntimeError("TLM adapter slot is not initialized")
        adapter = self.adapters[slot]
        if int(adapter.rank) != update_rank:
            raise ValueError("TLM adapter rank does not match requested update rank")
        weights = TestTimeLoRAWeights(
            lora_a=adapter.down.weight.detach().transpose(0, 1).contiguous().clone(),
            lora_b=adapter.up.weight.detach().transpose(0, 1).contiguous().clone(),
            alpha=export_alpha,
        )
        return TestTimeLoRAUpdate(
            adapter_key=adapter_key,
            lora_int_id=lora_int_id,
            rank=update_rank,
            scope=scope,
            target_modules={target_module_name: weights},
            request_ids=export_request_ids,
            step_id=step_id,
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

    def update_online(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor | None = None,
        *,
        logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        allow_scalar_fallback: bool = False,
    ) -> None:
        self.engine.update_online(
            hidden,
            scores,
            logits=logits,
            labels=labels,
            base_logits=base_logits,
            allow_scalar_fallback=allow_scalar_fallback,
        )

    def update_offline(
        self,
        hidden: torch.Tensor,
        scores: torch.Tensor | None = None,
        *,
        logits: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        base_logits: torch.Tensor | None = None,
        allow_scalar_fallback: bool = False,
    ) -> None:
        self.engine.update_offline(
            hidden,
            scores,
            logits=logits,
            labels=labels,
            base_logits=base_logits,
            allow_scalar_fallback=allow_scalar_fallback,
        )

    def flush_offline(self) -> None:
        self.engine.flush_offline()

    def read_stats(self) -> TLMStats:
        return self.engine.read_stats()

    def export_test_time_lora_update(
        self,
        *,
        adapter_key: str,
        lora_int_id: int,
        target_module_name: str,
        rank: int,
        scope: LoRAScope = "global",
        request_ids: tuple[str, ...] = (),
        step_id: int = -1,
        adapter_slot: int = 0,
        alpha: float = 1.0,
    ) -> TestTimeLoRAUpdate:
        return self.engine.export_test_time_lora_update(
            adapter_key=adapter_key,
            lora_int_id=lora_int_id,
            target_module_name=target_module_name,
            rank=rank,
            scope=scope,
            request_ids=request_ids,
            step_id=step_id,
            adapter_slot=adapter_slot,
            alpha=alpha,
        )
