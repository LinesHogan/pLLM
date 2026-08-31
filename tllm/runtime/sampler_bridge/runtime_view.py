#!/usr/bin/env python3
"""Runtime helpers for building same-step sampler views."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from typing import Protocol

import torch

from tllm.contracts.packed_rows import PackedSamplerRowBatch
from tllm.runtime.sampler_bridge.types import (
    NoSamplerSourceHidden,
    SamplerStepView,
)


class _PackedSamplerExecution(Protocol):
    @property
    def requires_packed_sampler_rows(self) -> bool: ...

    def packed_sampler_rows_for_step(
        self, step_id: int
    ) -> PackedSamplerRowBatch: ...


class _PackedSamplerRuntime(Protocol):
    @property
    def workload_execution(self) -> _PackedSamplerExecution | None: ...

    @property
    def event_step_id(self) -> int: ...


class _PackedSamplerRunner(Protocol):
    @property
    def model(self) -> torch.nn.Module: ...


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


def build_packed_sampler_step_view(
    *,
    runtime: _PackedSamplerRuntime,
    runner: _PackedSamplerRunner,
    model: torch.nn.Module,
    logits: torch.Tensor,
    sampling_metadata: _PackedSamplingMetadata | None,
) -> SamplerStepView | None:
    """Bind sampler logits to the execution's explicit packed-row links."""

    execution = runtime.workload_execution
    if execution is None or not execution.requires_packed_sampler_rows:
        return None
    step_id = int(runtime.event_step_id)
    sampler_rows = execution.packed_sampler_rows_for_step(step_id)
    row_count = len(sampler_rows.links)
    if logits.ndim != 2 or int(logits.shape[0]) < row_count:
        raise RuntimeError(
            "packed sampler logits do not cover the signed sampler rows"
        )
    identities = sampler_rows.identities
    return SamplerStepView(
        engine_step_id=step_id,
        phase="packed",
        logits=logits[:row_count],
        sampling_metadata=sampling_metadata,
        decode_count=row_count,
        request_ids=tuple(identity.request_id for identity in identities),
        prompt_idxs=tuple(identity.prompt_idx for identity in identities),
        sample_idxs=tuple(identity.sample_idx for identity in identities),
        prompt_idx_tensor=None,
        sample_idx_tensor=None,
        source_hidden=NoSamplerSourceHidden(),
        device=logits.device,
        model=model,
        runner=runner,
    )


def build_sampler_step_view(
    *,
    runtime: Any,
    runner: Any,
    model: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> SamplerStepView | None:
    packed = build_packed_sampler_step_view(
        runtime=runtime,
        runner=runner,
        model=model,
        logits=logits,
        sampling_metadata=sampling_metadata,
    )
    if packed is not None:
        return packed
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
