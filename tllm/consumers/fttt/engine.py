#!/usr/bin/env python3
"""FTTT engine, judge, and feedback machinery."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Real
import re
from typing import Protocol, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from tllm.contracts.test_time_lora import LoRAScope, TestTimeLoRAUpdate, TestTimeLoRAWeights
from tllm.consumers._common.alignment import LowRankAdapter, detached_train_input
from tllm.consumers.fttt.config import FTTTConsumerConfig
from tllm.runtime.generation_trace_runtime import CompletionTraceEntry


def _require_feedback_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be int")
    return value


def _require_feedback_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")
    return value


def _require_feedback_reward(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("reward must be real")
    return float(value)


def _require_feedback_int_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{field_name} must be tuple")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{field_name} must contain int")
        out.append(int(item))
    return tuple(out)


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


@dataclass(frozen=True)
class FeedbackRecord:
    prompt_idx: int
    feedback: str
    reward: float
    request_id: str = ""
    sample_idx: int = 0
    output_index: int = 0
    group_id: str = ""
    failed_text: str = ""
    failed_token_ids: tuple[int, ...] = ()
    feedback_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_idx", _require_feedback_int(self.prompt_idx, field_name="prompt_idx"))
        object.__setattr__(self, "feedback", _require_feedback_str(self.feedback, field_name="feedback"))
        object.__setattr__(self, "reward", _require_feedback_reward(self.reward))
        object.__setattr__(self, "request_id", _require_feedback_str(self.request_id, field_name="request_id"))
        object.__setattr__(self, "sample_idx", _require_feedback_int(self.sample_idx, field_name="sample_idx"))
        object.__setattr__(self, "output_index", _require_feedback_int(self.output_index, field_name="output_index"))
        object.__setattr__(self, "group_id", _require_feedback_str(self.group_id, field_name="group_id"))
        object.__setattr__(self, "failed_text", _require_feedback_str(self.failed_text, field_name="failed_text"))
        object.__setattr__(
            self,
            "failed_token_ids",
            _require_feedback_int_tuple(self.failed_token_ids, field_name="failed_token_ids"),
        )
        object.__setattr__(
            self,
            "feedback_token_ids",
            _require_feedback_int_tuple(self.feedback_token_ids, field_name="feedback_token_ids"),
        )


@dataclass(frozen=True)
class FTTTJudgeResult:
    is_correct: bool
    reward: float | None = None
    feedback_text: str = ""
    feedback_token_ids: tuple[int, ...] = ()
    label: str = ""
    judge: str = ""
    skip_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.is_correct, bool):
            raise TypeError("is_correct must be bool")
        reward = 1.0 if self.reward is None and self.is_correct else self.reward
        reward = 0.0 if reward is None else _require_feedback_reward(reward)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "feedback_text", _require_feedback_str(self.feedback_text, field_name="feedback_text"))
        object.__setattr__(
            self,
            "feedback_token_ids",
            _require_feedback_int_tuple(self.feedback_token_ids, field_name="feedback_token_ids"),
        )
        object.__setattr__(self, "label", _require_feedback_str(self.label, field_name="label"))
        object.__setattr__(self, "judge", _require_feedback_str(self.judge, field_name="judge"))
        object.__setattr__(self, "skip_reason", _require_feedback_str(self.skip_reason, field_name="skip_reason"))


class FTTTJudge(Protocol):
    def __call__(self, entry: CompletionTraceEntry) -> FTTTJudgeResult:
        ...


def _default_trace_judge(entry: CompletionTraceEntry) -> FTTTJudgeResult:
    text = str(getattr(entry, "text", "") or "").strip().lower()
    finish_reason = str(getattr(entry, "finish_reason", "") or "").strip().lower()
    negative_markers = ("incorrect", "wrong", "failed", "failure", "error")
    positive_markers = ("correct", "passed", "success", "solved")
    is_negative = any(marker in text for marker in negative_markers)
    is_positive = any(marker in text for marker in positive_markers) and not is_negative
    if is_positive:
        return FTTTJudgeResult(
            is_correct=True,
            reward=1.0,
            feedback_text="correct",
            judge="fttt-default-heuristic",
            skip_reason="correct-output",
        )
    if finish_reason and finish_reason != "stop":
        return FTTTJudgeResult(
            is_correct=False,
            reward=0.0,
            feedback_text=f"Generation did not finish cleanly: {finish_reason}.",
            judge="fttt-default-heuristic",
        )
    return FTTTJudgeResult(
        is_correct=False,
        reward=0.0,
        feedback_text="Your answer is incorrect.",
        judge="fttt-default-heuristic",
    )


@dataclass(frozen=True)
class FTTTInstanceState:
    retries: int = 0
    stopped: bool = False
    adapter_version: int = 0


@dataclass(frozen=True)
class _FTTTInstanceKey:
    prompt_idx: int
    request_id: str = ""


@dataclass(frozen=True)
class _FTTTLifecycleDecision:
    adaptable: bool
    next_state: FTTTInstanceState
    retry_action: str
    retry_reason: str
    retry_after_update: bool
    barrier_state: str
    barrier_reason: str
    ready_count: int


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
    adapter_version: int = 0


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
        self._adapter_version = 0
        self._instances: dict[_FTTTInstanceKey, FTTTInstanceState] = {}
        if int(config.hidden_size) > 0:
            self._build_adapter(int(config.hidden_size))

    @staticmethod
    def _detached_train_input(tensor: torch.Tensor) -> torch.Tensor:
        return detached_train_input(tensor)

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

    def _record_feature_text(self, record: FeedbackRecord) -> str:
        failed_token_text = " ".join(f"failed_tok_{token_id}" for token_id in record.failed_token_ids)
        feedback_token_text = " ".join(f"feedback_tok_{token_id}" for token_id in record.feedback_token_ids)
        return "\n".join(
            (
                f"feedback: {record.feedback}",
                f"failed_completion: {record.failed_text}",
                f"failed_tokens: {failed_token_text}",
                f"feedback_tokens: {feedback_token_text}",
            )
        )

    def _features_for_records(self, feedback: Sequence[FeedbackRecord], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack(
            [self._feedback_features(self._record_feature_text(record), device=device, dtype=dtype) for record in feedback],
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

    def _instance_key(self, *, prompt_idx: int, request_id: str = "", sample_idx: int = 0) -> _FTTTInstanceKey:
        _ = sample_idx
        request_id = str(request_id)
        if not request_id:
            return _FTTTInstanceKey(prompt_idx=int(prompt_idx))
        return _FTTTInstanceKey(prompt_idx=int(prompt_idx), request_id=request_id)

    def _instance_key_for_record(self, record: FeedbackRecord) -> _FTTTInstanceKey:
        return self._instance_key(
            prompt_idx=int(record.prompt_idx),
            request_id=record.request_id,
            sample_idx=int(record.sample_idx),
        )

    def _state_for_key(self, key: _FTTTInstanceKey) -> FTTTInstanceState:
        return self._instances.get(key, FTTTInstanceState())

    def _set_state_for_key(self, key: _FTTTInstanceKey, state: FTTTInstanceState) -> None:
        self._instances[key] = state

    def _aggregate_state_for_prompt(self, prompt_idx: int) -> FTTTInstanceState:
        prompt_idx = int(prompt_idx)
        legacy_key = self._instance_key(prompt_idx=prompt_idx)
        if legacy_key in self._instances:
            return self._instances[legacy_key]
        states = [state for key, state in self._instances.items() if int(key.prompt_idx) == prompt_idx]
        if not states:
            return FTTTInstanceState()
        return FTTTInstanceState(
            retries=max(int(state.retries) for state in states),
            stopped=any(bool(state.stopped) for state in states),
            adapter_version=max(int(state.adapter_version) for state in states),
        )

    def _lifecycle_decision(
        self,
        record: FeedbackRecord,
        state: FTTTInstanceState,
    ) -> _FTTTLifecycleDecision:
        if bool(state.stopped):
            return _FTTTLifecycleDecision(
                adaptable=False,
                next_state=state,
                retry_action="stop",
                retry_reason="instance-stopped",
                retry_after_update=False,
                barrier_state="blocked",
                barrier_reason="instance-stopped",
                ready_count=0,
            )
        if self._is_positive_feedback(record):
            next_state = FTTTInstanceState(
                retries=int(state.retries),
                stopped=True,
                adapter_version=int(state.adapter_version),
            )
            return _FTTTLifecycleDecision(
                adaptable=False,
                next_state=next_state,
                retry_action="accept",
                retry_reason="positive-feedback",
                retry_after_update=False,
                barrier_state="blocked",
                barrier_reason="positive-feedback",
                ready_count=0,
            )
        next_retries = int(state.retries) + 1
        retry_limit_reached = int(self.config.max_retries) > 0 and next_retries >= int(self.config.max_retries)
        next_state = FTTTInstanceState(
            retries=next_retries,
            stopped=retry_limit_reached,
            adapter_version=int(state.adapter_version),
        )
        return _FTTTLifecycleDecision(
            adaptable=True,
            next_state=next_state,
            retry_action="stop" if retry_limit_reached else "retry",
            retry_reason="retry-limit" if retry_limit_reached else "feedback-update",
            retry_after_update=not retry_limit_reached,
            barrier_state="ready",
            barrier_reason="feedback-update-ready",
            ready_count=1,
        )

    def _record_is_adaptable(self, record: FeedbackRecord) -> bool:
        key = self._instance_key_for_record(record)
        decision = self._lifecycle_decision(record, self._state_for_key(key))
        self._set_state_for_key(key, decision.next_state)
        if not decision.adaptable:
            self._skipped_feedback += 1
            return False
        return True

    def _mark_records_updated(self, feedback: Sequence[FeedbackRecord]) -> None:
        for record in feedback:
            key = self._instance_key_for_record(record)
            state = self._state_for_key(key)
            self._set_state_for_key(
                key,
                FTTTInstanceState(
                    retries=int(state.retries),
                    stopped=bool(state.stopped),
                    adapter_version=int(self._adapter_version),
                ),
            )

    def instance_state(self, prompt_idx: int, *, request_id: str = "", sample_idx: int = 0) -> FTTTInstanceState:
        if request_id:
            return self._state_for_key(
                self._instance_key(prompt_idx=int(prompt_idx), request_id=request_id, sample_idx=int(sample_idx))
            )
        return self._aggregate_state_for_prompt(int(prompt_idx))

    def reset_instance(self, prompt_idx: int, *, request_id: str = "", sample_idx: int = 0) -> None:
        prompt_idx = int(prompt_idx)
        if request_id:
            self._instances.pop(
                self._instance_key(prompt_idx=prompt_idx, request_id=request_id, sample_idx=int(sample_idx)),
                None,
            )
            return
        for key in tuple(self._instances):
            if int(key.prompt_idx) == prompt_idx:
                self._instances.pop(key, None)

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
        self._adapter_version += 1
        self._mark_records_updated(train_feedback)
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
            0.0 if self._state_for_key(self._instance_key_for_record(record)).stopped else 1.0
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
            adapter_version=int(self._adapter_version),
        )

    @property
    def adapter_version(self) -> int:
        return int(self._adapter_version)

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
        alpha: float = 1.0,
    ) -> TestTimeLoRAUpdate:
        update_rank = _require_lora_export_int(rank, field_name="rank")
        _require_lora_export_int(lora_int_id, field_name="lora_int_id")
        _require_lora_export_int(step_id, field_name="step_id")
        export_request_ids = _require_lora_export_request_ids(request_ids)
        export_alpha = _require_lora_export_real(alpha, field_name="alpha")
        if self.adapter is None:
            raise RuntimeError("FTTT adapter is not initialized")
        if int(self.adapter.rank) != update_rank:
            raise ValueError("FTTT adapter rank does not match requested update rank")
        weights = TestTimeLoRAWeights(
            lora_a=self.adapter.down.weight.detach().transpose(0, 1).contiguous().clone(),
            lora_b=self.adapter.up.weight.detach().transpose(0, 1).contiguous().clone(),
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
