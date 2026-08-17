#!/usr/bin/env python3
"""Torch reference backend for min-p exact candidate sampling."""

from __future__ import annotations

import os
from typing import Any

import torch

from tllm.runtime.sampler_bridge.min_p import apply_min_p, min_p_keep_mask
from tllm.runtime.sampler_bridge.truth import project_candidate_logits, select_candidate_pairs
from tllm.runtime.sampler_bridge.types import CandidateModifierState

# Fixed capacity for the sync-free compaction path used when min-p is applied
# natively upstream (vLLM >= 0.25). Rows with more survivors than the cap keep
# only the first `cap` found; "0" falls back to the exact `.nonzero()` path.
_COMPACT_CAP = int(os.getenv("TLLM_EXACT_COMPACT_CAP", "256") or "0")


def _sample_compacted_candidates(
    *,
    filtered: torch.Tensor,
    state: CandidateModifierState,
    cap: int,
) -> torch.Tensor | None:
    """Sync-free candidate sampling over the finite (post-filter) positions.

    Survivors are compacted on-device into fixed ``[rows, cap]`` buffers, the
    distiller is projected only on those candidates (batched bmm), and the
    draw is a gumbel argmax over the reweighted candidates — distributionally
    identical to sampling the reweighted full-vocab distribution.
    """
    from tllm.runtime.sampler_bridge.kernels.finite_compact_triton import compact_finite_ids

    rows = int(filtered.shape[0])
    row_ids = torch.arange(rows, device=filtered.device, dtype=torch.int32)
    result = compact_finite_ids(filtered, row_ids, cap=cap)
    if result is None:
        return None
    ids, _counts = result
    cap = int(ids.shape[1])
    valid = ids >= 0
    safe_ids = ids.clamp_min(0).to(torch.long)
    # Padded slot count, not the true survivor total (reading ``counts`` would
    # sync); diagnostics only.
    llm_vals = filtered.gather(1, safe_ids).to(torch.float32)
    if state.backend == "post_filter_dense_cache" and state.precomputed_dense_logits is not None:
        dense = state.precomputed_dense_logits.to(device=filtered.device, dtype=torch.float32)
        if state.pred_hidden_row_map is not None:
            dense_rows = state.pred_hidden_row_map.to(device=dense.device, dtype=torch.long)
        else:
            dense_rows = torch.arange(rows, device=dense.device, dtype=torch.long)
        distiller_vals = dense[dense_rows.unsqueeze(1), safe_ids.to(device=dense.device)]
    else:
        weight = state.lm_head_weight
        token_rows = weight.index_select(0, safe_ids.reshape(-1).to(device=weight.device, dtype=torch.long))
        token_rows = token_rows.view(rows, cap, -1)
        pred = state.pred_hidden
        if state.pred_hidden_row_map is not None:
            pred = pred.index_select(0, state.pred_hidden_row_map.to(device=pred.device, dtype=torch.long))
        distiller_vals = torch.bmm(token_rows, pred.to(device=filtered.device, dtype=token_rows.dtype).unsqueeze(-1)).squeeze(-1)
        if state.lm_head_bias is not None:
            distiller_vals = distiller_vals + state.lm_head_bias.index_select(
                0, safe_ids.reshape(-1).to(device=state.lm_head_bias.device, dtype=torch.long)
            ).view(rows, cap)
        distiller_vals = distiller_vals.to(torch.float32)
    beta = float(state.beta)
    candidate_logits = (1.0 + beta) * llm_vals - beta * distiller_vals.to(device=filtered.device)
    candidate_logits = torch.where(valid, candidate_logits, float("-inf"))
    noise = torch.empty_like(candidate_logits)
    noise.exponential_()
    scores = candidate_logits - noise.clamp_min(torch.finfo(torch.float32).tiny).log_()
    winners = scores.argmax(dim=-1)
    return safe_ids.gather(1, winners.unsqueeze(1)).view(-1)


def sample_rows_exact_candidates(
    *,
    logits: torch.Tensor,
    sampling_metadata: Any,
    state: CandidateModifierState,
    row_ids: torch.Tensor,
    all_rows: bool | None = None,
) -> torch.Tensor | None:
    """Sample directly from post-filter candidates without rebuilding full-vocab probs."""
    from tllm.runtime.sampler_bridge import exact_backend

    top_k_hint = getattr(sampling_metadata, "top_k", None)
    top_p_hint = getattr(sampling_metadata, "top_p", None)
    pure_min_p = top_k_hint is None and top_p_hint is None
    subset_logits, subset_state, top_k, top_p, min_p, generators = exact_backend._select_subset_inputs(
        logits=logits,
        sampling_metadata=sampling_metadata,
        state=state,
        row_ids=row_ids,
        all_rows=all_rows,
        clone_logits=not pure_min_p,
    )
    if generators:
        return None

    filtered = exact_backend.apply_top_k_top_p(subset_logits, top_k, top_p)
    if min_p is None and _COMPACT_CAP > 0:
        # vLLM >= 0.25: min-p was applied natively upstream as an
        # argmax-invariant logits processor, so survivors are exactly the
        # finite logits. The compacted path avoids the `.nonzero()` host
        # sync that stalls the decode launch pipeline every step.
        sampled = _sample_compacted_candidates(filtered=filtered, state=subset_state, cap=_COMPACT_CAP)
        if sampled is not None:
            exact_backend._record_candidate_stats(
                candidate_count=int(subset_logits.shape[0]) * _COMPACT_CAP,
                row_count=int(filtered.shape[0]),
            )
            return sampled
    if top_k is None and top_p is None and min_p is not None:
        candidate_row_ids, token_ids = min_p_keep_mask(filtered, min_p).nonzero(as_tuple=True)
    else:
        # Older versions pass ``min_p`` through the metadata; apply it here.
        if min_p is not None:
            filtered = apply_min_p(filtered, min_p)
        candidate_row_ids, token_ids = select_candidate_pairs(filtered)
    if candidate_row_ids.numel() <= 0:
        return filtered.argmax(dim=-1).to(dtype=torch.long)
    llm_candidate_logits = filtered[candidate_row_ids, token_ids]
    exact_backend._record_candidate_stats(candidate_count=candidate_row_ids.numel(), row_count=filtered.shape[0])
    if subset_state.backend == "post_filter_dense_cache" and subset_state.precomputed_dense_logits is not None:
        distiller_candidate_logits = exact_backend._gather_dense_candidate_logits(
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
    candidate_logits = ((1.0 + float(subset_state.beta)) * llm_candidate_logits) - (
        float(subset_state.beta) * distiller_candidate_logits.to(device=filtered.device, dtype=filtered.dtype)
    )
    noise = torch.empty_like(candidate_logits, dtype=torch.float32)
    noise.exponential_()
    scores = candidate_logits.to(dtype=torch.float32) - noise.log_()
    row_count = int(filtered.shape[0])
    row_scores = torch.full((row_count,), -float("inf"), device=filtered.device, dtype=torch.float32)
    row_scores.scatter_reduce_(0, candidate_row_ids, scores, reduce="amax", include_self=True)
    winner_mask = scores == row_scores.index_select(0, candidate_row_ids)
    sampled = torch.empty((row_count,), device=filtered.device, dtype=torch.long)
    sampled.scatter_(0, candidate_row_ids[winner_mask], token_ids[winner_mask].to(device=sampled.device, dtype=torch.long))
    return sampled
