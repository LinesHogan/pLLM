#!/usr/bin/env python3
"""Dependency-light boxed-answer verifier for TTRL reward grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tllm.consumers._common.alignment import normalize_answer_text


MISSING_BOXED_ANSWER = "missing-boxed-answer"
TTRL_DEFAULT_VERIFIER_JUDGE = "boxed_normalized_exact"


class TTRLVerifier(Protocol):
    @property
    def judge_name(self) -> str: ...

    @property
    def missing_reason(self) -> str: ...

    def label_for_completion(self, completion: str) -> str:
        """Return a normalized label, or an empty string when unverifiable."""
        ...


@dataclass(frozen=True)
class BoxedNormalizedVerifier:
    """Dependency-light verifier for exact matches on normalized boxed answers."""

    judge_name: str = TTRL_DEFAULT_VERIFIER_JUDGE
    missing_reason: str = MISSING_BOXED_ANSWER

    def label_for_completion(self, completion: str) -> str:
        return _extract_boxed_answer_label(completion)


def _last_boxed_answer_text(text: str) -> str:
    raw = str(text)
    idx = raw.rfind("\\boxed")
    if idx < 0:
        idx = raw.rfind("\\fbox")
    if idx < 0:
        return ""
    left_brace_idx = raw.find("{", idx)
    if left_brace_idx < 0:
        return ""
    depth = 0
    for pos in range(left_brace_idx, len(raw)):
        char = raw[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[left_brace_idx + 1 : pos]
    return ""


def _extract_boxed_answer_label(text: str) -> str:
    answer = _last_boxed_answer_text(str(text))
    if not answer:
        return ""
    return normalize_answer_text(answer)


DEFAULT_TTRL_VERIFIER: TTRLVerifier = BoxedNormalizedVerifier()
