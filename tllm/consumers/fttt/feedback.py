"""Typed FTTT feedback records and deterministic tensor construction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Protocol, Sequence

import torch
from torch.nn import functional as F

from tllm.runtime.generation_trace_runtime import CompletionTraceEntry


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class FTTTJudgeResult:
    is_correct: bool
    reward: float = 0.0
    feedback_text: str = ""
    feedback_token_ids: tuple[int, ...] = ()
    label: str = ""
    judge: str = ""
    skip_reason: str = ""


class FTTTJudge(Protocol):
    def __call__(self, entry: CompletionTraceEntry) -> FTTTJudgeResult: ...


def _default_trace_judge(entry: CompletionTraceEntry) -> FTTTJudgeResult:
    text = entry.text.strip().lower()
    finish_reason = entry.finish_reason.strip().lower()
    negative_markers = ("incorrect", "wrong", "failed", "failure", "error")
    positive_markers = ("correct", "passed", "success", "solved")
    is_negative = any(marker in text for marker in negative_markers)
    if any(marker in text for marker in positive_markers) and not is_negative:
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
            feedback_text=f"Generation did not finish cleanly: {finish_reason}.",
            judge="fttt-default-heuristic",
        )
    return FTTTJudgeResult(
        is_correct=False,
        feedback_text="Your answer is incorrect.",
        judge="fttt-default-heuristic",
    )


@dataclass(frozen=True, slots=True)
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
    features = torch.zeros(feature_size, device=device, dtype=dtype)
    tokens = re.findall(r"[a-z0-9_]+", text.lower()) or [text]
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], byteorder="little") % feature_size
        sign = 1.0 if int.from_bytes(digest[4:], byteorder="little") % 2 == 0 else -1.0
        features[bucket] += sign
    return F.normalize(features, p=2.0, dim=0, eps=1.0)


def record_feature_text(record: FeedbackRecord) -> str:
    failed_tokens = " ".join(
        f"failed_tok_{token_id}" for token_id in record.failed_token_ids
    )
    feedback_tokens = " ".join(
        f"feedback_tok_{token_id}" for token_id in record.feedback_token_ids
    )
    return "\n".join(
        (
            "feedback: " + record.feedback,
            "failed_completion: " + record.failed_text,
            "failed_tokens: " + failed_tokens,
            "feedback_tokens: " + feedback_tokens,
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
        tuple(
            feedback_features(
                record_feature_text(record),
                feature_size=feature_size,
                device=device,
                dtype=dtype,
            )
            for record in feedback
        )
    )


def rewards_column(
    feedback: Sequence[FeedbackRecord],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.tensor(
        tuple(record.reward for record in feedback),
        device=device,
        dtype=dtype,
    ).reshape(-1, 1)


def edit_inputs(
    hidden: torch.Tensor,
    feedback: Sequence[FeedbackRecord],
    *,
    feature_size: int,
) -> torch.Tensor:
    features = features_for_records(
        feedback,
        feature_size=feature_size,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    rewards = rewards_column(feedback, device=hidden.device, dtype=hidden.dtype)
    return torch.cat((hidden, features, rewards), dim=1)


def target_delta(
    hidden: torch.Tensor,
    feedback: Sequence[FeedbackRecord],
    *,
    feature_size: int,
    edit_scale: float,
) -> torch.Tensor:
    features = features_for_records(
        feedback,
        feature_size=feature_size,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    target = torch.zeros_like(hidden)
    hidden_size = hidden.shape[1]
    for column in range(features.shape[1]):
        target[:, column % hidden_size] += features[:, column]
    target = target / max(1.0, features.shape[1] // hidden_size + 1)
    rewards = rewards_column(feedback, device=hidden.device, dtype=hidden.dtype)
    strength = (1.0 - rewards).clamp(min=0.0) * edit_scale
    return target * strength


def is_positive_feedback(
    record: FeedbackRecord, *, positive_reward_threshold: float
) -> bool:
    if record.reward >= positive_reward_threshold:
        return True
    tokens = set(re.findall(r"[a-z0-9_]+", record.feedback.lower()))
    negative = {"incorrect", "wrong", "bad", "fail", "failed", "failure", "mistake", "error"}
    positive = {"correct", "right", "solved", "passed", "positive"}
    return bool(tokens & positive) and not bool(tokens & negative)


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
