#!/usr/bin/env python3
"""FTTT-inspired feedback edit consumer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import re
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


@dataclass(frozen=True)
class FTTTInstanceState:
    retries: int = 0
    stopped: bool = False


@dataclass
class FTTTConsumerConfig:
    consumer_id: str = "fttt"
    enabled: bool = True
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3
    max_queue_size: int = 128
    feedback_feature_size: int = 32
    positive_reward_threshold: float = 1.0
    max_retries: int = 0
    edit_scale: float = 1.0


@dataclass(frozen=True)
class FTTTStats:
    queued_feedback: int = 0
    dropped_feedback: int = 0
    feedback_count: int = 0
    update_count: int = 0
    loss_count: int = 0
    last_loss: float = 0.0
    skipped_feedback: int = 0
    stopped_instances: int = 0


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
        self._skipped_feedback = 0
        self._instances: dict[int, FTTTInstanceState] = {}
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
        self.adapter = LowRankAdapter(
            in_dim=self._edit_input_size(int(hidden_size)),
            out_dim=int(hidden_size),
            rank=int(self.config.rank),
        )
        self.optimizer = torch.optim.Adam(self.adapter.parameters(), lr=float(self.config.learning_rate))

    def _edit_input_size(self, hidden_size: int) -> int:
        return int(hidden_size) + max(1, int(self.config.feedback_feature_size)) + 1

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

    def _feedback_features(self, feedback: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        feature_size = max(1, int(self.config.feedback_feature_size))
        features = torch.zeros(feature_size, device=device, dtype=dtype)
        text = str(feedback).lower()
        tokens = re.findall(r"[a-z0-9_]+", text)
        if not tokens:
            tokens = [text]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], byteorder="little", signed=False) % feature_size
            sign = 1.0 if int.from_bytes(digest[4:], byteorder="little", signed=False) % 2 == 0 else -1.0
            features[bucket] += sign
        norm = features.norm().clamp_min(torch.tensor(1.0, device=device, dtype=dtype))
        return features / norm

    def _features_for_records(self, feedback: Sequence[FeedbackRecord], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack(
            [self._feedback_features(record.feedback, device=device, dtype=dtype) for record in feedback],
            dim=0,
        )

    def _edit_inputs(self, hidden: torch.Tensor, feedback: Sequence[FeedbackRecord]) -> torch.Tensor:
        if int(hidden.shape[0]) != len(feedback):
            raise RuntimeError("FTTT edit requires one feedback record per hidden row")
        features = self._features_for_records(feedback, device=hidden.device, dtype=hidden.dtype)
        rewards = torch.tensor(
            [float(record.reward) for record in feedback],
            device=hidden.device,
            dtype=hidden.dtype,
        ).reshape(-1, 1)
        return torch.cat([hidden, features, rewards], dim=1)

    def _target_delta(self, hidden: torch.Tensor, feedback: Sequence[FeedbackRecord]) -> torch.Tensor:
        features = self._features_for_records(feedback, device=hidden.device, dtype=hidden.dtype)
        target = torch.zeros_like(hidden)
        hidden_size = int(hidden.shape[1])
        feature_size = int(features.shape[1])
        for col in range(feature_size):
            target[:, col % hidden_size] += features[:, col]
        target = target / max(1.0, float(feature_size // hidden_size + 1))
        rewards = torch.tensor(
            [float(record.reward) for record in feedback],
            device=hidden.device,
            dtype=hidden.dtype,
        ).reshape(-1, 1)
        strength = (1.0 - rewards).clamp(min=0.0) * float(self.config.edit_scale)
        return target * strength

    def _is_positive_feedback(self, record: FeedbackRecord) -> bool:
        if float(record.reward) >= float(self.config.positive_reward_threshold):
            return True
        tokens = set(re.findall(r"[a-z0-9_]+", str(record.feedback).lower()))
        negative_markers = {"incorrect", "wrong", "bad", "fail", "failed", "failure", "mistake", "error"}
        positive_markers = {"correct", "right", "solved", "passed", "positive"}
        return bool(tokens & positive_markers) and not bool(tokens & negative_markers)

    def _state_for(self, prompt_idx: int) -> FTTTInstanceState:
        return self._instances.get(int(prompt_idx), FTTTInstanceState())

    def _set_state(self, prompt_idx: int, state: FTTTInstanceState) -> None:
        self._instances[int(prompt_idx)] = state

    def _record_is_adaptable(self, record: FeedbackRecord) -> bool:
        prompt_idx = int(record.prompt_idx)
        state = self._state_for(prompt_idx)
        if state.stopped:
            self._skipped_feedback += 1
            return False
        if self._is_positive_feedback(record):
            self._set_state(prompt_idx, FTTTInstanceState(retries=state.retries, stopped=True))
            self._skipped_feedback += 1
            return False
        next_retries = int(state.retries) + 1
        stopped = int(self.config.max_retries) > 0 and next_retries >= int(self.config.max_retries)
        self._set_state(prompt_idx, FTTTInstanceState(retries=next_retries, stopped=stopped))
        return True

    def instance_state(self, prompt_idx: int) -> FTTTInstanceState:
        return self._state_for(int(prompt_idx))

    def reset_instance(self, prompt_idx: int) -> None:
        self._instances.pop(int(prompt_idx), None)

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
        adaptable_indices = [idx for idx, record in enumerate(feedback) if self._record_is_adaptable(record)]
        if not adaptable_indices:
            return
        index = torch.tensor(adaptable_indices, device=hidden.device, dtype=torch.long)
        train_feedback = [feedback[idx] for idx in adaptable_indices]
        edit_input = self._edit_inputs(hidden_train.index_select(0, index), train_feedback)
        target = self._target_delta(hidden_train.index_select(0, index), train_feedback)
        pred = self.adapter(edit_input)
        loss = F.mse_loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._feedback_count += len(train_feedback)
        self._update_count += 1
        self._loss_count += 1
        self._last_loss = float(loss.detach().cpu())

    def edit_delta(self, hidden: torch.Tensor, feedback: Sequence[FeedbackRecord]) -> torch.Tensor:
        if not bool(self.config.enabled):
            return torch.zeros_like(hidden)
        if len(feedback) <= 0:
            return torch.zeros_like(hidden)
        self.ensure_shape(hidden)
        assert self.adapter is not None
        edit_input = self._edit_inputs(self._detached_train_input(hidden), feedback)
        with torch.no_grad():
            delta = self.adapter(edit_input)
        masks = [
            0.0 if self._state_for(record.prompt_idx).stopped else 1.0
            for record in feedback
        ]
        mask = torch.tensor(masks, device=hidden.device, dtype=hidden.dtype).reshape(-1, 1)
        return delta * mask

    def apply_edit_to_hidden(self, hidden: torch.Tensor, feedback: Sequence[FeedbackRecord]) -> torch.Tensor:
        return hidden + self.edit_delta(hidden, feedback)

    def apply_edit_to_logits(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        feedback: Sequence[FeedbackRecord],
        *,
        output_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        delta = self.edit_delta(hidden, feedback)
        if output_weight is not None:
            logit_delta = F.linear(delta, output_weight.to(device=delta.device, dtype=delta.dtype))
        else:
            if int(logits.shape[-1]) != int(delta.shape[-1]):
                raise RuntimeError("FTTT logits edit without output_weight requires logits and hidden to share width")
            logit_delta = delta
        return logits + logit_delta.to(device=logits.device, dtype=logits.dtype)

    def read_stats(self, *, queued_feedback: int, dropped_feedback: int) -> FTTTStats:
        return FTTTStats(
            queued_feedback=int(queued_feedback),
            dropped_feedback=int(dropped_feedback),
            feedback_count=int(self._feedback_count),
            update_count=int(self._update_count),
            loss_count=int(self._loss_count),
            last_loss=float(self._last_loss),
            skipped_feedback=int(self._skipped_feedback),
            stopped_instances=sum(1 for state in self._instances.values() if state.stopped),
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

    def edit_delta(self, hidden: torch.Tensor, feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord]) -> torch.Tensor:
        records = list(feedback)
        return self.engine.edit_delta(hidden, records)

    def apply_edit_to_hidden(
        self,
        hidden: torch.Tensor,
        feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord],
    ) -> torch.Tensor:
        records = list(feedback)
        return self.engine.apply_edit_to_hidden(hidden, records)

    def apply_edit_to_logits(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        feedback: Sequence[FeedbackRecord] | Iterable[FeedbackRecord],
        *,
        output_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        records = list(feedback)
        return self.engine.apply_edit_to_logits(logits, hidden, records, output_weight=output_weight)

    def instance_state(self, prompt_idx: int) -> FTTTInstanceState:
        return self.engine.instance_state(prompt_idx)

    def reset_instance(self, prompt_idx: int) -> None:
        self.engine.reset_instance(prompt_idx)

    def read_stats(self) -> FTTTStats:
        return self.engine.read_stats(queued_feedback=len(self._queue), dropped_feedback=self._dropped_feedback)
