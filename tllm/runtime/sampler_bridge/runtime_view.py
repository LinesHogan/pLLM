#!/usr/bin/env python3
"""Runtime helpers for building same-step sampler views."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from typing import Protocol

import torch

from tllm.runtime.sampler_bridge.types import (
    SamplerStepView,
)


class _PackedSamplingMetadata(Protocol):
    @property
    def all_greedy(self) -> bool: ...

    @property
    def all_random(self) -> bool: ...

    @property
    def top_k(self) -> torch.Tensor | None: ...

    @property
    def top_p(self) -> torch.Tensor | None: ...

    @property
    def min_p(self) -> torch.Tensor | None: ...

    @property
    def generators(self) -> Mapping[int, torch.Generator]: ...


def build_sampler_step_view(
    *,
    runtime: Any,
    runner: Any,
    model: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> SamplerStepView | None:
    decode_count = int(getattr(runtime, "decode_count", 0) or 0)
    if logits.ndim != 2 or decode_count <= 0 or int(logits.shape[0]) != decode_count:
        return None
    source_path = str(getattr(runtime, "source_resolved_path", "") or "").strip()
    if not source_path:
        return None
    source_hidden = getattr(runtime, "tap_decode_hidden", {}).get(source_path)
    if not isinstance(source_hidden, torch.Tensor):
        return None
    if int(source_hidden.shape[0]) < decode_count:
        return None
    prompt_idx_tensor = getattr(runtime, "decode_prompt_idx_tensor", None)
    sample_idx_tensor = getattr(runtime, "decode_sample_idx_tensor", None)
    if isinstance(prompt_idx_tensor, torch.Tensor) and int(prompt_idx_tensor.numel()) >= decode_count:
        prompt_idx_tensor = prompt_idx_tensor[:decode_count]
    else:
        prompt_idx_tensor = None
    if isinstance(sample_idx_tensor, torch.Tensor) and int(sample_idx_tensor.numel()) >= decode_count:
        sample_idx_tensor = sample_idx_tensor[:decode_count]
    else:
        sample_idx_tensor = None
    prompt_idxs = tuple(int(x) for x in list(getattr(runtime, "decode_prompt_idxs", []))[:decode_count])
    sample_idxs = tuple(int(x) for x in list(getattr(runtime, "decode_sample_idxs", []))[:decode_count])
    raw_request_ids = list(getattr(runtime, "decode_request_ids", []))
    request_ids = tuple(str(x) for x in raw_request_ids[:decode_count])
    if len(prompt_idxs) != decode_count:
        return None
    if len(sample_idxs) != decode_count:
        return None
    if len(request_ids) != decode_count:
        return None
    return SamplerStepView(
        engine_step_id=int(getattr(runtime, "event_step_id", 0) or 0),
        phase="decode",
        logits=logits,
        sampling_metadata=sampling_metadata,
        decode_count=decode_count,
        request_ids=request_ids,
        prompt_idxs=prompt_idxs,
        sample_idxs=sample_idxs,
        prompt_idx_tensor=prompt_idx_tensor,
        sample_idx_tensor=sample_idx_tensor,
        source_hidden=source_hidden[:decode_count],
        device=logits.device,
        model=model,
        runner=runner,
    )
