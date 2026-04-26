#!/usr/bin/env python3
"""Runtime helpers for completed generation traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CompletionTraceEntry:
    request_id: str
    prompt_idx: int
    sample_idx: int
    output_index: int
    token_ids: tuple[int, ...]
    text: str
    finish_reason: str
    group_id: str


@dataclass(frozen=True)
class GenerationTraceBatch:
    entries: tuple[CompletionTraceEntry, ...]


def _candidate_sample_idx(*, candidate: Any, candidate_index: int, output_count: int, sample_idx_base: int) -> int:
    explicit_index = getattr(candidate, "index", None)
    if explicit_index is not None:
        return int(explicit_index)
    if int(output_count) == 1:
        return int(sample_idx_base)
    return int(candidate_index)


def _token_ids_tuple(token_ids: Any) -> tuple[int, ...]:
    if token_ids is None:
        return ()
    if hasattr(token_ids, "detach") and callable(getattr(token_ids, "detach")):
        token_ids = token_ids.detach().cpu().reshape(-1).tolist()
    return tuple(int(token_id) for token_id in token_ids)


def build_generation_trace_from_outputs(
    outputs: Any,
    *,
    resolve_prompt_sample_fn: Callable[[str], tuple[int, int]],
    group_id: str = "",
) -> GenerationTraceBatch:
    entries: list[CompletionTraceEntry] = []
    if outputs is None:
        return GenerationTraceBatch(entries=())

    for out in outputs:
        request_id = getattr(out, "request_id", None)
        if request_id is None:
            continue
        request_id_s = str(request_id)
        prompt_idx, sample_idx_base = resolve_prompt_sample_fn(request_id_s)
        if int(prompt_idx) < 0:
            continue
        candidates = tuple(getattr(out, "outputs", None) or ())
        for candidate_index, candidate in enumerate(candidates):
            token_ids = _token_ids_tuple(getattr(candidate, "token_ids", None))
            entries.append(
                CompletionTraceEntry(
                    request_id=request_id_s,
                    prompt_idx=int(prompt_idx),
                    sample_idx=_candidate_sample_idx(
                        candidate=candidate,
                        candidate_index=int(candidate_index),
                        output_count=len(candidates),
                        sample_idx_base=int(sample_idx_base),
                    ),
                    output_index=int(candidate_index),
                    token_ids=token_ids,
                    text=str(getattr(candidate, "text", "") or ""),
                    finish_reason=str(getattr(candidate, "finish_reason", "") or ""),
                    group_id=str(group_id),
                )
            )
    return GenerationTraceBatch(entries=tuple(entries))
