"""FTTT's judged feedback as numbers: each failed completion's hashed feedback features and its reward.

FTTTTrainer turns queued feedback records into these rows; they are the targets of the edit's training batch.
"""

from __future__ import annotations

import re
from hashlib import blake2b
from math import sqrt

from pllm.workloads.fttt.types import FTTTFeedbackRecord


def feedback_features(text: str, *, feature_size: int) -> list[float]:
    """Signed token hashes of `text` in `feature_size` buckets, scaled to unit length when longer than 1."""
    features = [0.0] * feature_size
    tokens = re.findall(r"[a-z0-9_]+", text.lower()) or (text,)
    for token in tokens:
        digest = blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], byteorder="little") % feature_size
        features[bucket] += 1.0 if int.from_bytes(digest[4:], byteorder="little") % 2 == 0 else -1.0
    norm = max(1.0, sqrt(sum(value * value for value in features)))
    return [value / norm for value in features]


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


def feedback_rows(records: tuple[FTTTFeedbackRecord, ...], *, feature_size: int) -> list[list[float]]:
    """One row per record: its `feature_size` feedback features, then its reward."""
    return [
        [*feedback_features(record_feature_text(record), feature_size=feature_size), record.reward]
        for record in records
    ]


__all__ = ["feedback_features", "feedback_rows", "record_feature_text"]
