#!/usr/bin/env python3
"""Generate wrappers that preserve request->prompt mapping during capture."""

from __future__ import annotations

from typing import Any, Callable, Sequence

from tllm.runtime.trajectory_store import apply_prompt_contexts
from tllm.runtime.generation_trace_runtime import build_generation_trace_from_outputs
from tllm.runtime.vllm_patch.port_runtime_hooks import dispatch_post_generate_port_bundles
from tllm.contracts.request_identity import resolve_prompt_sample


def _resolve_prompt_sample_for_req_id(runtime: Any, request_id: str) -> tuple[int, int]:
    prompt_map = getattr(runtime, "reqid_to_promptidx", {}) or {}
    sample_map = getattr(runtime, "reqid_to_sampleidx", {}) or {}
    return resolve_prompt_sample(
        request_id,
        prompt_indices=prompt_map,
        sample_indices=sample_map,
    )


def make_mapping_add_request(
    *,
    runtime: Any,
    plane: Any,
    orig_add_request: Callable[..., Any],
    request_prompt_indices: Sequence[int] | None,
    request_sample_indices: Sequence[int] | None,
) -> Callable[..., Any]:
    """Wrap LLMEngine.add_request so every request records its mapping.

    The mapping is recorded in the parent (for post-generate trace building)
    AND pushed to the EngineCore child via the control plane BEFORE the
    request itself is submitted. Utility RPCs and ADD requests share one FIFO
    input socket, so the child processes the mapping before the request can
    be scheduled — no race.
    """
    counter = {"i": 0}
    runtime.reqid_to_promptidx = {}
    runtime.reqid_to_sampleidx = {}

    def _wrapped_add_request(request_id, prompt, p, *args, **kwargs):
        req_i = int(counter["i"])
        if request_prompt_indices is not None and req_i < len(request_prompt_indices):
            prompt_idx = int(request_prompt_indices[req_i])
        else:
            prompt_idx = req_i
        if request_sample_indices is not None and req_i < len(request_sample_indices):
            sample_idx = int(request_sample_indices[req_i])
        else:
            sample_idx = 0
        counter["i"] += 1
        runtime.reqid_to_promptidx[str(request_id)] = prompt_idx
        runtime.reqid_to_sampleidx[str(request_id)] = sample_idx
        if plane is not None:
            plane.push_request_mapping([(str(request_id), prompt_idx, sample_idx)])
        return orig_add_request(request_id, prompt, p, *args, **kwargs)

    return _wrapped_add_request


def run_generate_with_request_mapping(
    *,
    runtime: Any,
    llm: Any,
    prompts: Any,
    params: Any,
    request_prompt_indices: Any = None,
    request_sample_indices: Any = None,
    generation_group_id: str = "",
    use_tqdm: bool | None = None,
    dispatch_post_generate: bool = True,
):
    """Run one generate call while tracking request->prompt index mapping."""
    from tllm.runtime.mp.control import ControlPlane

    orig_add_request = llm.llm_engine.add_request
    _wrapped_add_request = make_mapping_add_request(
        runtime=runtime,
        plane=ControlPlane.try_bind(llm),
        orig_add_request=orig_add_request,
        request_prompt_indices=request_prompt_indices,
        request_sample_indices=request_sample_indices,
    )

    effective_prompts = apply_prompt_contexts(
        runtime,
        prompts,
        request_prompt_indices=request_prompt_indices,
    )

    llm.llm_engine.add_request = _wrapped_add_request  # type: ignore[assignment]
    try:
        generate_kwargs = {} if use_tqdm is None else {"use_tqdm": bool(use_tqdm)}
        outputs = llm.generate(effective_prompts, params, **generate_kwargs)
    finally:
        llm.llm_engine.add_request = orig_add_request  # type: ignore[assignment]
    if dispatch_post_generate:
        runtime.last_generation_trace = build_generation_trace_from_outputs(
            outputs,
            resolve_prompt_sample_fn=lambda request_id: _resolve_prompt_sample_for_req_id(runtime, str(request_id)),
            group_id=str(generation_group_id),
        )
        dispatch_post_generate_port_bundles(runtime=runtime, runner=llm)
    return outputs
