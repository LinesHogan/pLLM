#!/usr/bin/env python3
"""FTTT feedback records, judge, and feature/target construction (pure Python).

Everything here is CPU-side control logic: what feedback means, how it is
hashed into features, and which target delta a record implies. The tensor
training on those features lives in ``program.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Real
import re
from typing import Protocol, Sequence

import torch

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
    text = str(entry.text or "").strip().lower()
    finish_reason = str(entry.finish_reason or "").strip().lower()
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


def feedback_features(
    text: str,
    *,
    feature_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    feature_size = max(1, int(feature_size))
    features = torch.zeros(feature_size, device=device, dtype=dtype)
    tokens = re.findall(r"[a-z0-9_]+", str(text).lower())
    if not tokens:
        tokens = [str(text)]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], byteorder="little", signed=False) % feature_size
        sign = 1.0 if int.from_bytes(digest[4:], byteorder="little", signed=False) % 2 == 0 else -1.0
        features[bucket] += sign
    norm = features.norm().clamp_min(torch.tensor(1.0, device=device, dtype=dtype))
    return features / norm


def record_feature_text(record: FeedbackRecord) -> str:
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


def features_for_records(
    feedback: Sequence[FeedbackRecord],
    *,
    feature_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack(
        [
            feedback_features(
                record_feature_text(record),
                feature_size=feature_size,
                device=device,
                dtype=dtype,
            )
            for record in feedback
        ],
        dim=0,
    )


def rewards_column(feedback: Sequence[FeedbackRecord], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(
        [float(record.reward) for record in feedback],
        device=device,
        dtype=dtype,
    ).reshape(-1, 1)


def edit_inputs(
    hidden: torch.Tensor,
    feedback: Sequence[FeedbackRecord],
    *,
    feature_size: int,
) -> torch.Tensor:
    if int(hidden.shape[0]) != len(feedback):
        raise RuntimeError("FTTT edit requires one feedback record per hidden row")
    features = features_for_records(
        feedback, feature_size=feature_size, device=hidden.device, dtype=hidden.dtype
    )
    return torch.cat([hidden, features, rewards_column(feedback, device=hidden.device, dtype=hidden.dtype)], dim=1)


def target_delta(
    hidden: torch.Tensor,
    feedback: Sequence[FeedbackRecord],
    *,
    feature_size: int,
    edit_scale: float,
) -> torch.Tensor:
    features = features_for_records(
        feedback, feature_size=feature_size, device=hidden.device, dtype=hidden.dtype
    )
    target = torch.zeros_like(hidden)
    hidden_size = int(hidden.shape[1])
    for col in range(int(features.shape[1])):
        target[:, col % hidden_size] += features[:, col]
    target = target / max(1.0, float(int(features.shape[1]) // hidden_size + 1))
    rewards = rewards_column(feedback, device=hidden.device, dtype=hidden.dtype)
    strength = (1.0 - rewards).clamp(min=0.0) * float(edit_scale)
    return target * strength


def is_positive_feedback(record: FeedbackRecord, *, positive_reward_threshold: float) -> bool:
    if float(record.reward) >= float(positive_reward_threshold):
        return True
    tokens = set(re.findall(r"[a-z0-9_]+", str(record.feedback).lower()))
    negative_markers = {"incorrect", "wrong", "bad", "fail", "failed", "failure", "mistake", "error"}
    positive_markers = {"correct", "right", "solved", "passed", "positive"}
    return bool(tokens & positive_markers) and not bool(tokens & negative_markers)


__all__ = [
    "FTTTInstanceState",
    "FTTTJudge",
    "FTTTJudgeResult",
    "FeedbackRecord",
    "_default_trace_judge",
    "edit_inputs",
    "features_for_records",
    "is_positive_feedback",
    "record_feature_text",
    "target_delta",
]
