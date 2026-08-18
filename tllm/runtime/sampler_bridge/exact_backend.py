#!/usr/bin/env python3
"""Exact correctness backend for distiller sampler intervention."""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from tllm.runtime.sampler_bridge.min_p import apply_min_p, min_p_keep_mask
from tllm.runtime.sampler_bridge.backends.dense_cache import gather_dense_candidate_logits as _gather_dense_candidate_logits
from tllm.runtime.sampler_bridge.truth import (
    apply_candidate_intervention,
    project_candidate_logits,
    select_candidate_pairs,
)
from tllm.runtime.sampler_bridge.types import SamplerModifierState

_MINP_TOPK_CANDIDATE_CAP = int(os.getenv("TLLM_EXACT_MINP_TOPK_CANDIDATES", "0") or "0")


def _runtime_state() -> Any | None:
    try:
        from tllm.runtime import residual_runtime

        return residual_runtime.RUNTIME
    except Exception:
        return None


def _record_candidate_stats(*, candidate_count: int, row_count: int) -> None:
    runtime = _runtime_state()
    if runtime is None:
        return
    precompute = runtime.sampler_precompute
    precompute.candidate_sample_count += 1
    precompute.candidate_token_count += int(candidate_count)
    precompute.candidate_row_count += int(row_count)
    precompute.candidate_max_count = max(precompute.candidate_max_count, int(candidate_count))


def _record_selected_candidate_stats(*, token_ids: torch.Tensor, row_count: int) -> None:
    _record_candidate_stats(candidate_count=int(token_ids.numel()), row_count=int(row_count))


def _select_sampling_tensor_rows(tensor: torch.Tensor | None, row_ids: torch.Tensor) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.index_select(0, row_ids.to(device=tensor.device, dtype=torch.long))


def _row_ids_cover_all_rows(*, logits: torch.Tensor, row_ids: torch.Tensor) -> bool:
    if int(row_ids.numel()) != int(logits.shape[0]):
        return False
    expected = torch.arange(int(logits.shape[0]), device=row_ids.device, dtype=torch.long)
    return bool(torch.equal(row_ids.to(device=expected.device, dtype=torch.long), expected))


def _select_generators(generators: dict[int, torch.Generator], row_ids: torch.Tensor) -> dict[int, torch.Generator]:
    if not generators:
        return {}
    selected: dict[int, torch.Generator] = {}
    for new_i, row_i in enumerate(row_ids.cpu().tolist()):
        generator = generators.get(int(row_i))
        if generator is not None:
            selected[int(new_i)] = generator
    return selected


def _subset_modifier_state(state: SamplerModifierState, row_ids: torch.Tensor) -> SamplerModifierState:
    pos_tensor = torch.searchsorted(
        state.affected_row_ids.to(device=row_ids.device, dtype=torch.long),
        row_ids.to(device=row_ids.device, dtype=torch.long),
    ).to(device=state.pred_hidden.device, dtype=torch.long)
    precomputed_dense_logits = None
    if state.precomputed_dense_logits is not None:
        precomputed_dense_logits = state.precomputed_dense_logits.index_select(
            0,
            pos_tensor.to(device=state.precomputed_dense_logits.device, dtype=torch.long),
        )
    return SamplerModifierState(
        beta=state.beta,
        backend=state.backend,
        affected_row_ids=row_ids,
        pred_hidden=state.pred_hidden.index_select(0, pos_tensor),
        lm_head_weight=state.lm_head_weight,
        lm_head_bias=state.lm_head_bias,
        precomputed_dense_logits=precomputed_dense_logits,
        pred_hidden_row_map=state.pred_hidden_row_map.index_select(0, pos_tensor.to(device=state.pred_hidden_row_map.device, dtype=torch.long))
        if state.pred_hidden_row_map is not None
        else None,
    )


def _select_subset_inputs(
    *,
    logits: torch.Tensor,
    sampling_metadata: Any,
    state: SamplerModifierState,
    row_ids: torch.Tensor,
    all_rows: bool | None = None,
    clone_logits: bool = True,
    ) -> tuple[torch.Tensor, SamplerModifierState, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[int, torch.Generator]]:
    if all_rows is None:
        all_rows = _row_ids_cover_all_rows(logits=logits, row_ids=row_ids)
    if all_rows:
        subset_logits = logits.clone() if clone_logits else logits
        subset_state = state
        top_k = getattr(sampling_metadata, "top_k", None)
        top_p = getattr(sampling_metadata, "top_p", None)
        min_p = getattr(sampling_metadata, "min_p", None)
        generators = getattr(sampling_metadata, "generators", {})
    else:
        subset_logits = logits.index_select(0, row_ids)
        if clone_logits:
            subset_logits = subset_logits.clone()
        subset_state = _subset_modifier_state(state, row_ids)
        top_k = _select_sampling_tensor_rows(getattr(sampling_metadata, "top_k", None), row_ids)
        top_p = _select_sampling_tensor_rows(getattr(sampling_metadata, "top_p", None), row_ids)
        min_p = _select_sampling_tensor_rows(getattr(sampling_metadata, "min_p", None), row_ids)
        generators = _select_generators(getattr(sampling_metadata, "generators", {}), row_ids)
    return subset_logits, subset_state, top_k, top_p, min_p, generators


def build_modified_logits_exact(
    *,
    logits: torch.Tensor,
    sampling_metadata: Any,
    state: SamplerModifierState,
    row_ids: torch.Tensor,
    greedy: bool,
    all_rows: bool | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Generator]]:
    subset_logits, subset_state, top_k, top_p, min_p, generators = _select_subset_inputs(
        logits=logits,
        sampling_metadata=sampling_metadata,
        state=state,
        row_ids=row_ids,
        all_rows=all_rows,
    )
    if greedy:
        filtered = apply_min_p(apply_top_k_top_p(subset_logits, top_k, top_p), min_p)
        candidate_row_ids, token_ids = select_candidate_pairs(filtered)
        _record_selected_candidate_stats(
            token_ids=token_ids,
            row_count=int(subset_logits.shape[0]),
        )
        if subset_state.backend == "post_filter_dense_cache" and subset_state.precomputed_dense_logits is not None:
            distiller_candidate_logits = _gather_dense_candidate_logits(
                state=subset_state,
                row_ids=candidate_row_ids,
                token_ids=token_ids,
                dtype=filtered.dtype,
                device=filtered.device,
            )
        else:
            distiller_candidate_logits = project_candidate_logits(
                pred_hidden=subset_state.pred_hidden,
                row_ids=candidate_row_ids,
                token_ids=token_ids,
                lm_head_weight=subset_state.lm_head_weight,
                lm_head_bias=subset_state.lm_head_bias,
                pred_hidden_row_map=subset_state.pred_hidden_row_map,
            )
        return (
            apply_candidate_intervention(
                logits=filtered,
                row_ids=candidate_row_ids,
                token_ids=token_ids,
                distiller_candidate_logits=distiller_candidate_logits,
                beta=subset_state.beta,
                in_place=True,
            ),
            generators,
        )

    filtered = apply_min_p(apply_top_k_top_p(subset_logits, top_k, top_p), min_p)
    candidate_row_ids, token_ids = select_candidate_pairs(filtered)
    _record_selected_candidate_stats(
        token_ids=token_ids,
        row_count=int(subset_logits.shape[0]),
    )
    if subset_state.backend == "post_filter_dense_cache" and subset_state.precomputed_dense_logits is not None:
        distiller_candidate_logits = _gather_dense_candidate_logits(
            state=subset_state,
            row_ids=candidate_row_ids,
            token_ids=token_ids,
            dtype=filtered.dtype,
            device=filtered.device,
        )
    else:
        distiller_candidate_logits = project_candidate_logits(
            pred_hidden=subset_state.pred_hidden,
            row_ids=candidate_row_ids,
            token_ids=token_ids,
            lm_head_weight=subset_state.lm_head_weight,
            lm_head_bias=subset_state.lm_head_bias,
            pred_hidden_row_map=subset_state.pred_hidden_row_map,
        )
    return (
        apply_candidate_intervention(
            logits=filtered,
            row_ids=candidate_row_ids,
            token_ids=token_ids,
            distiller_candidate_logits=distiller_candidate_logits,
            beta=subset_state.beta,
            in_place=True,
        ),
        generators,
    )


def sample_rows_exact_candidates(
    *,
    logits: torch.Tensor,
    sampling_metadata: Any,
    state: SamplerModifierState,
    row_ids: torch.Tensor,
    all_rows: bool | None = None,
) -> torch.Tensor | None:
    """Sample directly from post-filter candidates without rebuilding full-vocab probs."""
    top_k_hint = getattr(sampling_metadata, "top_k", None)
    top_p_hint = getattr(sampling_metadata, "top_p", None)
    if top_k_hint is None and top_p_hint is None and _MINP_TOPK_CANDIDATE_CAP > 0:
        subset_logits, subset_state, _top_k, _top_p, min_p, generators = _select_subset_inputs(
            logits=logits,
            sampling_metadata=sampling_metadata,
            state=state,
            row_ids=row_ids,
            all_rows=all_rows,
            clone_logits=False,
        )
        if generators or min_p is None:
            return None
        from tllm.runtime.sampler_bridge.backends import fixed_topk_minp

        return fixed_topk_minp.sample_fixed_topk_min_p_candidates(
            logits=subset_logits,
            min_p=min_p,
            state=subset_state,
            cap=_MINP_TOPK_CANDIDATE_CAP,
            record_candidate_stats=lambda candidate_count, row_count: _record_candidate_stats(
                candidate_count=candidate_count,
                row_count=row_count,
            ),
        )

    from tllm.runtime.sampler_bridge.backends import minp_exact_torch

    return minp_exact_torch.sample_rows_exact_candidates(
        logits=logits,
        sampling_metadata=sampling_metadata,
        state=state,
        row_ids=row_ids,
        all_rows=all_rows,
    )
