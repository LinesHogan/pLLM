"""Pure deterministic FTTT feedback features and edit targets."""

from __future__ import annotations

import re
from hashlib import blake2b

import torch
from torch.nn import functional as F

from pllm.workloads.fttt.types import FTTTFeedbackRecord


def feedback_features(
    text: str,
    *,
    feature_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    features = torch.zeros(feature_size, device=device, dtype=dtype)
    tokens = re.findall(r"[a-z0-9_]+", text.lower()) or (text,)
    for token in tokens:
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], byteorder="little") % feature_size
        sign = 1.0 if int.from_bytes(digest[4:], byteorder="little") % 2 == 0 else -1.0
        features[bucket] += sign
    return F.normalize(features, p=2.0, dim=0, eps=1.0)


def record_feature_text(record: FTTTFeedbackRecord) -> str:
    failed_tokens = " ".join(f"failed_tok_{token_id}" for token_id in record.completion.token_ids)
    feedback_tokens = " ".join(f"feedback_tok_{token_id}" for token_id in record.feedback_token_ids)
    return "\n".join(
        (
            "feedback: " + record.feedback_text,
            "failed_completion: " + record.completion.text,
            "failed_tokens: " + failed_tokens,
            "feedback_tokens: " + feedback_tokens,
        )
    )


def features_for_records(
    records: tuple[FTTTFeedbackRecord, ...],
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
            for record in records
        )
    )


def edit_training_tensors(
    hidden: torch.Tensor,
    records: tuple[FTTTFeedbackRecord, ...],
    *,
    feature_size: int,
    edit_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = features_for_records(
        records,
        feature_size=feature_size,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    rewards = torch.tensor(
        tuple(record.reward for record in records),
        device=hidden.device,
        dtype=hidden.dtype,
    ).reshape(-1, 1)
    target = torch.zeros_like(hidden)
    hidden_size = int(hidden.shape[1])
    for column in range(int(features.shape[1])):
        target[:, column % hidden_size] += features[:, column]
    target = target / max(1.0, int(features.shape[1]) // hidden_size + 1)
    return torch.cat((hidden, features, rewards), dim=1), target * ((1.0 - rewards).clamp(min=0.0) * edit_scale)


__all__ = [
    "edit_training_tensors",
    "feedback_features",
    "features_for_records",
    "record_feature_text",
]
