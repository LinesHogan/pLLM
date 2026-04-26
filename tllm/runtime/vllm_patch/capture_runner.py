#!/usr/bin/env python3
"""Generate wrappers that preserve request->prompt mapping during capture."""

from __future__ import annotations

from typing import Any

from tllm.runtime.generation_trace_runtime import build_generation_trace_from_outputs


def _resolve_prompt_sample_for_req_id(runtime: Any, request_id: str) -> tuple[int, int]:
    request_id_s = str(request_id)
    prompt_map = getattr(runtime, "reqid_to_promptidx", {}) or {}
    sample_map = getattr(runtime, "reqid_to_sampleidx", {}) or {}

    prompt_idx = prompt_map.get(request_id_s)
    if prompt_idx is not None:
        return int(prompt_idx), int(sample_map.get(request_id_s, 0))

    if "_" in request_id_s:
        maybe_sample_idx, parent_request_id = request_id_s.split("_", 1)
        if maybe_sample_idx.isdigit():
            parent_prompt_idx = prompt_map.get(parent_request_id)
            if parent_prompt_idx is not None:
                return int(parent_prompt_idx), int(maybe_sample_idx)
    return -1, -1


def run_generate_with_request_mapping(
    *,
    runtime: Any,
    llm: Any,
    prompts: Any,
    params: Any,
    request_prompt_indices: Any = None,
    request_sample_indices: Any = None,
    generation_group_id: str = "",
):
    """Run one generate call while tracking request->prompt index mapping."""
    orig_add_request = llm.llm_engine.add_request
    counter = {"i": 0}
    runtime.reqid_to_promptidx = {}
    runtime.reqid_to_sampleidx = {}

    def _wrapped_add_request(request_id, prompt, p, *args, **kwargs):
        req_i = int(counter["i"])
        if request_prompt_indices is not None and req_i < len(request_prompt_indices):
            runtime.reqid_to_promptidx[str(request_id)] = int(request_prompt_indices[req_i])
        else:
            runtime.reqid_to_promptidx[str(request_id)] = req_i
        if request_sample_indices is not None and req_i < len(request_sample_indices):
            runtime.reqid_to_sampleidx[str(request_id)] = int(request_sample_indices[req_i])
        else:
            runtime.reqid_to_sampleidx[str(request_id)] = 0
        counter["i"] += 1
        return orig_add_request(request_id, prompt, p, *args, **kwargs)

    llm.llm_engine.add_request = _wrapped_add_request  # type: ignore[assignment]
    try:
        outputs = llm.generate(prompts, params)
    finally:
        llm.llm_engine.add_request = orig_add_request  # type: ignore[assignment]
    runtime.last_generation_trace = build_generation_trace_from_outputs(
        outputs,
        resolve_prompt_sample_fn=lambda request_id: _resolve_prompt_sample_for_req_id(runtime, str(request_id)),
        group_id=str(generation_group_id),
    )
    return outputs
