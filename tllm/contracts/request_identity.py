#!/usr/bin/env python3
"""Stable request identity resolved at the vLLM/private-API boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestIdentity:
    request_id: str
    prompt_idx: int
    sample_idx: int


@dataclass(frozen=True, slots=True)
class RequestFinishedBatch:
    requests: tuple[RequestIdentity, ...]


def _external_request_id(request_id: str, prompt_indices: Mapping[str, int]) -> str | None:
    if request_id in prompt_indices:
        return request_id
    # vLLM 0.27 EngineCore internal ids append "-<random8>". Exact lookup
    # above wins for legitimate external ids that already have this shape.
    prefix, separator, suffix = request_id.rpartition("-")
    if separator and len(suffix) == 8 and suffix.isalnum() and prefix in prompt_indices:
        return prefix
    return None


def resolve_request_identity(
    request_id: str,
    *,
    prompt_indices: Mapping[str, int],
    sample_indices: Mapping[str, int] | None = None,
) -> RequestIdentity | None:
    """Resolve direct, parallel-child, and vLLM-internal request ids."""
    sample_map: Mapping[str, int] = {} if sample_indices is None else sample_indices
    external = _external_request_id(request_id, prompt_indices)
    if external is not None:
        return RequestIdentity(
            request_id=external,
            prompt_idx=int(prompt_indices[external]),
            sample_idx=int(sample_map.get(external, 0)),
        )

    sample_text, separator, parent_id = request_id.partition("_")
    if separator and sample_text.isdigit():
        external_parent = _external_request_id(parent_id, prompt_indices)
        if external_parent is not None:
            return RequestIdentity(
                request_id=external_parent,
                prompt_idx=int(prompt_indices[external_parent]),
                sample_idx=int(sample_text),
            )
    return None


def resolve_prompt_sample(
    request_id: str,
    *,
    prompt_indices: Mapping[str, int],
    sample_indices: Mapping[str, int] | None = None,
) -> tuple[int, int]:
    identity = resolve_request_identity(
        request_id,
        prompt_indices=prompt_indices,
        sample_indices=sample_indices,
    )
    if identity is None:
        return -1, -1
    return identity.prompt_idx, identity.sample_idx
