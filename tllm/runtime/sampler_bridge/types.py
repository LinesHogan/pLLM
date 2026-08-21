#!/usr/bin/env python3
"""Runtime-internal sampler step view and sampling-metadata adapters.

The public modifier contract vocabulary lives in ``tllm.ports.sampler`` and is
re-exported here for the bridge's internal users.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tllm.ports.sampler import (
    SAMPLER_BACKEND_CHOICES,
    CandidateModifierRequest,
    CandidateModifierState,
    CandidateSampleResult,
    CandidateSamplingMode,
    CanonicalSamplerBackend,
    SamplerBackend,
    SamplerFilterSpec,
    SamplerModifierState,
    TemperatureMode,
)

__all__ = [
    "SAMPLER_BACKEND_CHOICES",
    "CandidateModifierRequest",
    "CandidateModifierState",
    "CandidateSampleResult",
    "CandidateSamplingMode",
    "CanonicalSamplerBackend",
    "SamplerBackend",
    "SamplerFilterSpec",
    "SamplerModifierState",
    "SamplerStepView",
    "TemperatureMode",
    "filter_spec_from_sampling_metadata",
    "normalize_sampler_backend",
]

_SAMPLER_BACKEND_ALIASES: dict[str, CanonicalSamplerBackend] = {
    "post_filter_exact": "post_filter_exact",
    "post_filter_exact_minp": "post_filter_exact",
    "post_filter_exact_torch": "post_filter_exact",
    "pre_filter_dense": "pre_filter_dense",
    "post_filter_dense_cache": "post_filter_dense_cache",
}


def normalize_sampler_backend(backend: str | None) -> CanonicalSamplerBackend:
    key = str(backend or "post_filter_exact").strip() or "post_filter_exact"
    try:
        return _SAMPLER_BACKEND_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported distiller sampler backend: {key}") from exc


def _metadata_temperature_mode(sampling_metadata: Any) -> TemperatureMode:
    if bool(getattr(sampling_metadata, "all_greedy", False)):
        return "all_greedy"
    if bool(getattr(sampling_metadata, "all_random", False)):
        return "all_random"
    return "mixed"


def _normalize_top_k(top_k: torch.Tensor | None, *, vocab_size: int | None) -> torch.Tensor | None:
    if top_k is None:
        return None
    if top_k.numel() <= 0:
        return None
    if top_k.device.type != "cpu":
        return top_k
    if bool(torch.all(top_k <= 0)):
        return None
    if vocab_size is not None:
        vocab = int(vocab_size)
        if bool(torch.all((top_k <= 0) | (top_k >= vocab))):
            return None
    return top_k


def _normalize_top_p(top_p: torch.Tensor | None) -> torch.Tensor | None:
    if top_p is None:
        return None
    if top_p.numel() <= 0:
        return None
    if top_p.device.type != "cpu":
        return top_p
    if bool(torch.all(top_p >= 1.0)):
        return None
    return top_p


def _normalize_min_p(min_p: torch.Tensor | None) -> torch.Tensor | None:
    if min_p is None:
        return None
    if min_p.numel() <= 0:
        return None
    if min_p.device.type != "cpu":
        return min_p
    if bool(torch.all(min_p <= 0.0)):
        return None
    return min_p


@dataclass(frozen=True)
class SamplerStepView:
    engine_step_id: int
    phase: str
    logits: torch.Tensor
    sampling_metadata: Any
    decode_count: int
    request_ids: tuple[str, ...]
    prompt_idxs: tuple[int, ...]
    sample_idxs: tuple[int, ...]
    prompt_idx_tensor: torch.Tensor | None
    sample_idx_tensor: torch.Tensor | None
    source_hidden: torch.Tensor
    device: torch.device
    model: Any
    runner: Any


def filter_spec_from_sampling_metadata(
    sampling_metadata: Any,
    *,
    vocab_size: int | None = None,
) -> SamplerFilterSpec:
    generators = getattr(sampling_metadata, "generators", {}) or {}
    return SamplerFilterSpec(
        top_k=_normalize_top_k(getattr(sampling_metadata, "top_k", None), vocab_size=vocab_size),
        top_p=_normalize_top_p(getattr(sampling_metadata, "top_p", None)),
        min_p=_normalize_min_p(getattr(sampling_metadata, "min_p", None)),
        temperature_mode=_metadata_temperature_mode(sampling_metadata),
        has_generators=bool(generators),
    )
