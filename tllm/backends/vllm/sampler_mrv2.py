#!/usr/bin/env python3
"""Instance-scoped sampler interceptor for vLLM 0.25+ MRv2.

The MRv2 sampler (``vllm.v1.worker.gpu.sample.sampler.Sampler``) is a plain
callable. ``sample()`` applies bias/penalties/bad-words/temperature/min-p
via ``apply_sampling_params(skip_top_k_top_p=True)``, then filters with
top-k/top-p (either fused inside FlashInfer or via ``apply_top_k_top_p``)
and draws with ``gumbel_sample``.

Post-filter modifiers require the "filter first, intervene on survivors" stage, which does
not exist inside the fused FlashInfer kernel. The interceptor therefore:

* leaves the instance untouched when no provider is active (bit-for-bit the
  upstream path, FlashInfer included);
* when a provider is active, forces the non-FlashInfer path for that step,
  applies top-k/top-p explicitly, reweights only the surviving candidates
  with ``(1+beta)*logit - beta*distiller_logit``, and draws with the same
  ``gumbel_sample`` call (same seeds/positions, so RNG consumption matches
  the unmodified non-FlashInfer path);
* for ``pre_filter_dense`` providers, applies the intervention to the raw
  logits *before* ``apply_sampling_params`` — the ``softmax((p - λh)/T)``
  pipeline order — on a clone, so penalties, temperature and min-p all
  operate on the intervened distribution.

The pre-draw fence is dispatched before ``provider.prepare_step`` consumes
any Region output (side-stream join precedes consumption), matching the
generic bridge and the MRv1 wrapper.

The class is never patched globally; the wrapper is installed on the
sampler instance owned by the bound runner.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from tllm.backends.protocol import runner_model_module
from tllm.runtime.sampler_bridge.types import CandidateModifierState, SamplerStepView
from tllm.runtime.sampler_bridge.provider import validate_prepared_intervention
from tllm.runtime.step_session import StepState, get_step_session
from tllm.runtime.vllm_patch.sampler_patch import (
    _notify_after_sample,
    _resolve_provider,
    dispatch_sampler_pre_draw,
)

_MARK_ATTR = "_tllm_mrv2_sampler_wrapped"
_SAMPLING_EPS = 1e-5
# Fixed capacity for the sync-free survivor-compaction path. Rows with more
# surviving (post-filter) tokens than the cap are reweighted only on the first
# `cap` survivors found; "0" disables the candidate path and forces the dense
# full-vocab fallback.
_INTERVENTION_CANDIDATE_CAP = int(os.getenv("TLLM_MRV2_INTERVENTION_CANDIDATE_CAP", "256") or "0")


def _build_mrv2_step_view(*, runtime: Any, runner: Any, logits: torch.Tensor) -> SamplerStepView | None:
    from tllm.runtime.sampler_bridge.runtime_view import (
        build_packed_sampler_step_view,
    )

    execution = runtime.workload_execution
    if execution is not None and execution.requires_packed_sampler_rows:
        return build_packed_sampler_step_view(
            runtime=runtime,
            runner=runner,
            model=runner_model_module(runner),
            logits=logits,
            sampling_metadata=None,
        )
    decode_count = int(getattr(runtime, "decode_count", 0) or 0)
    if logits.ndim != 2 or decode_count <= 0 or int(logits.shape[0]) < decode_count:
        return None
    source_path = str(getattr(runtime, "source_resolved_path", "") or "").strip()
    if not source_path:
        return None
    source_hidden = getattr(runtime, "tap_decode_hidden", {}).get(source_path)
    if not isinstance(source_hidden, torch.Tensor) or int(source_hidden.shape[0]) < decode_count:
        return None
    prompt_idx_tensor = getattr(runtime, "decode_prompt_idx_tensor", None)
    if isinstance(prompt_idx_tensor, torch.Tensor) and int(prompt_idx_tensor.numel()) >= decode_count:
        prompt_idx_tensor = prompt_idx_tensor[:decode_count]
    else:
        prompt_idx_tensor = None
    prompt_idxs = tuple(int(x) for x in list(getattr(runtime, "decode_prompt_idxs", []))[:decode_count])
    if len(prompt_idxs) != decode_count:
        return None
    sample_idx_tensor = getattr(runtime, "decode_sample_idx_tensor", None)
    if isinstance(sample_idx_tensor, torch.Tensor) and int(sample_idx_tensor.numel()) >= decode_count:
        sample_idx_tensor = sample_idx_tensor[:decode_count]
    else:
        sample_idx_tensor = None
    sample_idxs = tuple(int(x) for x in list(getattr(runtime, "decode_sample_idxs", []))[:decode_count])
    if len(sample_idxs) != decode_count:
        return None
    raw_request_ids = list(getattr(runtime, "decode_request_ids", []))
    request_ids = tuple(str(x) for x in raw_request_ids[:decode_count]) if len(raw_request_ids) >= decode_count else ()
    return SamplerStepView(
        engine_step_id=int(getattr(runtime, "event_step_id", 0) or 0),
        phase="decode",
        logits=logits[:decode_count],
        sampling_metadata=None,
        decode_count=decode_count,
        request_ids=request_ids,
        prompt_idxs=prompt_idxs,
        sample_idxs=sample_idxs,
        prompt_idx_tensor=prompt_idx_tensor,
        sample_idx_tensor=sample_idx_tensor,
        source_hidden=source_hidden[:decode_count],
        device=logits.device,
        model=getattr(runner, "model", None),
        runner=runner,
    )


def _apply_intervention(
    *,
    filtered_logits: torch.Tensor,
    state: CandidateModifierState,
) -> torch.Tensor:
    """Reweight surviving candidates: (1+beta)*logit - beta*distiller_logit.

    Rows outside ``affected_row_ids`` and masked (-inf) entries are left
    untouched, so the intervention can never resurrect filtered-out tokens.
    """
    affected = state.affected_row_ids.to(device=filtered_logits.device, dtype=torch.long)
    if int(affected.numel()) <= 0:
        return filtered_logits
    beta = float(state.beta)
    if beta == 0.0:
        return filtered_logits

    if state.pred_hidden_row_map is not None:
        pred_rows = state.pred_hidden_row_map.to(device=filtered_logits.device, dtype=torch.long).index_select(
            0, affected
        )
    else:
        pred_rows = affected

    dense = state.precomputed_dense_logits
    if _INTERVENTION_CANDIDATE_CAP > 0:
        compacted = _reweight_candidates_syncfree(
            filtered_logits=filtered_logits,
            affected=affected,
            pred_rows=pred_rows,
            state=state,
            beta=beta,
            cap=_INTERVENTION_CANDIDATE_CAP,
        )
        if compacted is not None:
            return compacted

    if dense is not None:
        dense_rows = dense.to(device=filtered_logits.device, dtype=torch.float32).index_select(0, pred_rows)
    else:
        pred_hidden = state.pred_hidden.to(device=filtered_logits.device).index_select(0, pred_rows)
        weight = state.lm_head_weight
        dense_rows = F.linear(pred_hidden.to(dtype=weight.dtype), weight, state.lm_head_bias).to(torch.float32)

    sub = filtered_logits.index_select(0, affected).to(torch.float32)
    keep = torch.isfinite(sub)
    reweighted = torch.where(keep, (1.0 + beta) * sub - beta * dense_rows, sub)
    out = filtered_logits.clone()
    out[affected] = reweighted.to(dtype=filtered_logits.dtype)
    return out


def _reweight_candidates_syncfree(
    *,
    filtered_logits: torch.Tensor,
    affected: torch.Tensor,
    pred_rows: torch.Tensor,
    state: CandidateModifierState,
    beta: float,
    cap: int,
) -> torch.Tensor | None:
    """Reweight post-filter survivors without any host sync.

    Survivors (finite logits) are compacted on-device into a fixed-capacity
    buffer; only those positions receive the ``(1+beta)*x - beta*d`` update,
    applied as an additive correction ``beta*(x - d)``. Returns None when the
    compaction kernel is unavailable (caller falls back to the dense path).
    """
    from tllm.runtime.sampler_bridge.kernels.finite_compact_triton import compact_finite_ids

    result = compact_finite_ids(filtered_logits, affected, cap=cap)
    if result is None:
        return None
    ids, _counts = result
    valid = ids >= 0
    safe_ids = ids.clamp_min(0).to(torch.long)

    llm_vals = filtered_logits[affected.unsqueeze(1), safe_ids].to(torch.float32)
    distiller_vals = _candidate_distiller_logits(
        state=state,
        pred_rows=pred_rows,
        safe_ids=safe_ids,
        device=filtered_logits.device,
    )
    correction = torch.where(
        valid,
        beta * (llm_vals - distiller_vals.to(device=filtered_logits.device)),
        torch.zeros((), device=filtered_logits.device, dtype=torch.float32),
    )
    # Additive index_put is safe for duplicate padded positions: they carry a
    # zero correction, and -inf entries stay -inf under +0.
    filtered_logits.index_put_(
        (affected.unsqueeze(1).expand(-1, int(ids.shape[1])), safe_ids),
        correction.to(dtype=filtered_logits.dtype),
        accumulate=True,
    )
    return filtered_logits


def _candidate_distiller_logits(
    *,
    state: CandidateModifierState,
    pred_rows: torch.Tensor,
    safe_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Project distiller logits for ``[rows, cap]`` candidate token ids."""
    rows = int(safe_ids.shape[0])
    cap = int(safe_ids.shape[1])
    dense = state.precomputed_dense_logits
    if dense is not None:
        dense_dev = dense.to(device=device, dtype=torch.float32)
        return dense_dev[
            pred_rows.to(device=dense_dev.device, dtype=torch.long).unsqueeze(1),
            safe_ids.to(device=dense_dev.device),
        ]
    weight = state.lm_head_weight
    token_rows = weight.index_select(0, safe_ids.reshape(-1).to(device=weight.device, dtype=torch.long))
    token_rows = token_rows.view(rows, cap, -1)
    pred = state.pred_hidden.to(device=device).index_select(
        0, pred_rows.to(device=state.pred_hidden.device, dtype=torch.long)
    )
    distiller_vals = torch.bmm(token_rows, pred.to(dtype=token_rows.dtype).unsqueeze(-1)).squeeze(-1)
    if state.lm_head_bias is not None:
        distiller_vals = distiller_vals + state.lm_head_bias.index_select(
            0, safe_ids.reshape(-1).to(device=state.lm_head_bias.device, dtype=torch.long)
        ).view(rows, cap)
    return distiller_vals.to(torch.float32)


def _gumbel_over_candidates(
    *,
    candidate_logits: torch.Tensor,
    valid: torch.Tensor,
    safe_ids: torch.Tensor,
    sampler: Any,
    affected: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    row_count: int,
) -> torch.Tensor:
    """Draw one token per affected row from ``[rows, cap]`` candidates.

    Gumbel-max over the survivors is distributionally identical to sampling
    the reweighted full-vocab distribution; greedy rows (temperature == 0)
    take the reweighted argmax, mirroring upstream.
    """
    candidate_logits = torch.where(valid, candidate_logits, float("-inf"))
    temp_rows = sampler.sampling_states.temperature.gpu.index_select(
        0,
        expanded_idx_mapping.to(device=candidate_logits.device, dtype=torch.long).index_select(0, affected),
    )
    random_mask = (temp_rows >= _SAMPLING_EPS).unsqueeze(1)
    noise = torch.empty_like(candidate_logits)
    noise.exponential_()
    gumbel = -noise.clamp_min(torch.finfo(torch.float32).tiny).log_()
    scores = torch.where(random_mask, candidate_logits + gumbel, candidate_logits)
    winners = scores.argmax(dim=-1)
    sampled_candidates = safe_ids.gather(1, winners.unsqueeze(1)).view(-1)
    sampled = torch.empty((int(row_count),), device=candidate_logits.device, dtype=torch.long)
    sampled[affected] = sampled_candidates
    return sampled


def _record_mrv2_candidate_stats(
    runtime: Any,
    *,
    candidate_count: int,
    row_count: int,
) -> None:
    precompute = runtime.sampler_precompute
    precompute.candidate_sample_count += 1
    precompute.candidate_token_count += int(candidate_count)
    precompute.candidate_row_count += int(row_count)
    precompute.candidate_max_count = max(
        int(precompute.candidate_max_count),
        int(candidate_count),
    )
    precompute.candidate_kernel_triton_count += 1


def _sample_post_filter_minp_fused(
    *,
    runtime: Any,
    sampler: Any,
    logits: torch.Tensor,
    state: CandidateModifierState,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping_np: Any,
    pos: torch.Tensor,
) -> torch.Tensor | None:
    """Fused min-p + compaction + candidate draw, skipping all full-vocab fp32
    work (the upcast copy, the in-place mask write, and the sampling read).

    Only valid when min-p is the sole active logits transform for every row:
    penalties/bias/bad-words mutate logits outside this kernel, and top-k/top-p
    need their own filter. Those steps fall back to the generic paths.
    """
    if _INTERVENTION_CANDIDATE_CAP <= 0:
        return None
    beta = float(state.beta)
    if beta == 0.0:
        return None
    affected = state.affected_row_ids.to(device=logits.device, dtype=torch.long)
    if int(affected.numel()) <= 0 or int(affected.numel()) != int(logits.shape[0]):
        return None
    import numpy as np

    states = sampler.sampling_states
    if bool(np.any(sampler.logit_bias_state.use_logit_bias[idx_mapping_np])):
        return None
    if bool(np.any(sampler.penalties_state.use_penalty[idx_mapping_np])):
        return None
    if bool(np.any(sampler.bad_words_state.num_bad_words.np[idx_mapping_np] > 0)):
        return None
    if not bool(np.all(states.min_p.np[idx_mapping_np] > 0.0)):
        return None
    top_k, top_p = states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
    if top_k is not None or top_p is not None:
        return None

    from tllm.runtime.sampler_bridge.kernels.minp_compact_triton import (
        gumbel_argmax_candidates,
        minp_compact_candidates,
    )

    result = minp_compact_candidates(
        logits,
        expanded_idx_mapping,
        states.temperature.gpu,
        states.min_p.gpu,
        cap=_INTERVENTION_CANDIDATE_CAP,
    )
    if result is None:
        return None
    ids, vals, _counts = result
    safe_ids = ids.clamp_min(0).to(torch.long)

    if state.pred_hidden_row_map is not None:
        pred_rows = state.pred_hidden_row_map.to(device=logits.device, dtype=torch.long).index_select(0, affected)
    else:
        pred_rows = affected
    distiller_vals = _candidate_distiller_logits(
        state=state,
        pred_rows=pred_rows,
        safe_ids=safe_ids,
        device=logits.device,
    )
    sampled = gumbel_argmax_candidates(
        vals,
        distiller_vals,
        ids,
        beta=beta,
        temperature=states.temperature.gpu,
        seeds=states.seeds.gpu,
        pos=pos,
        affected=affected,
        expanded_idx_mapping=expanded_idx_mapping,
        row_count=int(logits.shape[0]),
    )
    _record_mrv2_candidate_stats(
        runtime,
        candidate_count=int(ids.numel()),
        row_count=int(affected.numel()),
    )
    return sampled


def _sample_post_filter_candidates(
    *,
    runtime: Any,
    sampler: Any,
    filtered_logits: torch.Tensor,
    state: CandidateModifierState,
    expanded_idx_mapping: torch.Tensor,
) -> torch.Tensor | None:
    """Sample directly from the compacted survivor set (no full-vocab writes).

    Returns None when the fast path does not apply (cap disabled, compaction
    kernel unavailable, or the step has rows outside ``affected_row_ids``).
    Sampling among the survivors with fresh gumbel noise is distributionally
    identical to sampling the reweighted full-vocab distribution, because
    filtered-out positions carry -inf either way. Greedy (temperature == 0)
    rows take the argmax of the reweighted candidates, mirroring upstream.
    """
    if _INTERVENTION_CANDIDATE_CAP <= 0:
        return None
    affected = state.affected_row_ids.to(device=filtered_logits.device, dtype=torch.long)
    if int(affected.numel()) != int(filtered_logits.shape[0]):
        # Mixed prefill/decode steps still need the full-vocab path for the
        # unaffected rows; use the in-place reweight fallback instead.
        return None
    if int(affected.numel()) <= 0:
        return None
    beta = float(state.beta)
    if beta == 0.0:
        return None

    from tllm.runtime.sampler_bridge.kernels.finite_compact_triton import compact_finite_ids

    result = compact_finite_ids(filtered_logits, affected, cap=_INTERVENTION_CANDIDATE_CAP)
    if result is None:
        return None
    ids, _counts = result
    cap = int(ids.shape[1])
    valid = ids >= 0
    safe_ids = ids.clamp_min(0).to(torch.long)

    if state.pred_hidden_row_map is not None:
        pred_rows = state.pred_hidden_row_map.to(device=filtered_logits.device, dtype=torch.long).index_select(
            0, affected
        )
    else:
        pred_rows = affected

    llm_vals = filtered_logits[affected.unsqueeze(1), safe_ids].to(torch.float32)
    distiller_vals = _candidate_distiller_logits(
        state=state,
        pred_rows=pred_rows,
        safe_ids=safe_ids,
        device=filtered_logits.device,
    )
    candidate_logits = (1.0 + beta) * llm_vals - beta * distiller_vals
    sampled = _gumbel_over_candidates(
        candidate_logits=candidate_logits,
        valid=valid,
        safe_ids=safe_ids,
        sampler=sampler,
        affected=affected,
        expanded_idx_mapping=expanded_idx_mapping,
        row_count=int(filtered_logits.shape[0]),
    )
    _record_mrv2_candidate_stats(
        runtime,
        candidate_count=int(ids.numel()),
        row_count=int(affected.numel()),
    )
    return sampled


def make_mrv2_sample_wrapper(*, sampler: Any, runtime: Any, runner: Any, orig_sample: Any):
    def _sample(
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: Any,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        return_logprobs: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        provider = _resolve_provider(runtime)
        if provider is None or not provider.is_active():
            return orig_sample(
                logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                return_logprobs=return_logprobs,
            )
        session = get_step_session(runner)
        if session is None or session.state in (
            StepState.CREATED,
            StepState.BACKEND_WARMUP,
        ):
            return orig_sample(
                logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                return_logprobs=return_logprobs,
            )

        import os as _os
        import time as _time

        _trace = _os.getenv("TLLM_TRACE_MRV2_SAMPLER", "") == "1"
        _tp0 = _time.perf_counter() if _trace else 0.0
        view = _build_mrv2_step_view(runtime=runtime, runner=runner, logits=logits)
        # The parameter-read fence must publish *before* the provider reads
        # any Region output: dispatch joins side-stream work onto the main
        # stream, and provider consumption after the join observes complete
        # results.  The reverse order raced side Regions signed before
        # sampler.pre_draw (mirrors sample_with_optional_modifier).
        dispatch_sampler_pre_draw(
            runtime=runtime,
            runner=runner,
            device=logits.device,
        )
        state = provider.prepare_step(view) if view is not None else None
        if _trace:
            from tllm.runtime.residual_runtime import record_path_hotspot_cpu

            record_path_hotspot_cpu("mrv2_sampler.prepare_step_cpu", (_time.perf_counter() - _tp0) * 1000.0)
        if state is None or int(state.affected_row_ids.numel()) <= 0 or view is None:
            return orig_sample(
                logits,
                expanded_idx_mapping,
                idx_mapping_np,
                pos,
                input_ids,
                expanded_local_pos,
                return_logprobs=return_logprobs,
            )
        validate_prepared_intervention(provider, state)
        runtime.sampler_precompute.intervention_state_count += 1

        from vllm.config.model import PROCESSED_LOGPROBS_MODES
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        _t0 = _time.perf_counter() if _trace else 0.0

        def _hotspot(name: str, t0: float) -> float:
            if not _trace:
                return 0.0
            from tllm.runtime.residual_runtime import record_path_hotspot_cpu

            t1 = _time.perf_counter()
            record_path_hotspot_cpu(name, (t1 - t0) * 1000.0)
            return t1

        needs_processed = return_logprobs and sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES
        if state.backend != "pre_filter_dense" and not needs_processed:
            sampled = _sample_post_filter_minp_fused(
                runtime=runtime,
                sampler=sampler,
                logits=logits,
                state=state,
                expanded_idx_mapping=expanded_idx_mapping,
                idx_mapping_np=idx_mapping_np,
                pos=pos,
            )
            if sampled is not None:
                _t3 = _hotspot("mrv2_sampler.filter_intervene_cpu", _t0)
                _hotspot("mrv2_sampler.gumbel_cpu", _t3)
                _notify_after_sample(provider=provider, view=view, sampled_token_ids=sampled)
                return sampled, logits

        if state.backend == "pre_filter_dense":
            # Pre-filter dense semantics: reweight the RAW logits before the
            # sampling pipeline — softmax((p - λh)/T), not softmax(p/T - λh).
            # Penalties, temperature and min-p then apply to the intervened
            # distribution.  The raw logits alias the runner's tensor, so the
            # intervention always works on a clone.
            from tllm.runtime.sampler_bridge.bridge import (
                apply_pre_filter_dense_logits,
            )

            logits = apply_pre_filter_dense_logits(logits=logits, state=state)
        _ti = _hotspot("mrv2_sampler.pre_pipeline_intervene_cpu", _t0)

        processed_logits = sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )
        _t1 = _hotspot("mrv2_sampler.apply_sampling_params_cpu", _ti)
        top_k, top_p = sampler.sampling_states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        _t2 = _hotspot("mrv2_sampler.get_top_k_top_p_cpu", _t1)

        if state.backend == "pre_filter_dense":
            # Intervention already applied to the raw logits above; only the
            # top-k/top-p filter remains before the draw.
            filtered = apply_top_k_top_p(processed_logits, top_k, top_p)
        else:
            # post-filter: filter the base logits first, then reweight only
            # the surviving candidates. FlashInfer fuses filter+draw and has
            # no hook point between them, so this step always takes the
            # explicit path.
            filtered = apply_top_k_top_p(processed_logits, top_k, top_p)
            if not needs_processed:
                sampled = _sample_post_filter_candidates(
                    runtime=runtime,
                    sampler=sampler,
                    filtered_logits=filtered,
                    state=state,
                    expanded_idx_mapping=expanded_idx_mapping,
                )
                if sampled is not None:
                    _t3 = _hotspot("mrv2_sampler.filter_intervene_cpu", _t2)
                    _hotspot("mrv2_sampler.gumbel_cpu", _t3)
                    _notify_after_sample(provider=provider, view=view, sampled_token_ids=sampled)
                    return sampled, processed_logits
            if filtered is logits:
                filtered = filtered.clone()
            filtered = _apply_intervention(filtered_logits=filtered, state=state)
        _t3 = _hotspot("mrv2_sampler.filter_intervene_cpu", _t2)

        sampled = gumbel_sample(
            filtered,
            expanded_idx_mapping,
            sampler.sampling_states.temperature.gpu,
            sampler.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
            use_fp64=sampler.use_fp64_gumbel,
        )
        _hotspot("mrv2_sampler.gumbel_cpu", _t3)
        _notify_after_sample(provider=provider, view=view, sampled_token_ids=sampled)
        if return_logprobs and sampler.logprobs_mode in PROCESSED_LOGPROBS_MODES:
            return sampled, filtered
        return sampled, processed_logits

    return _sample


def bind_runner_sampler_mrv2(*, runtime: Any, runner: Any) -> None:
    sampler = getattr(runner, "sampler", None)
    if sampler is None:
        return
    setattr(runner, "_tllm_runtime", runtime)
    setattr(sampler, "_tllm_runtime", runtime)
    setattr(sampler, "_tllm_runner", runner)
    if getattr(sampler, _MARK_ATTR, False):
        return
    orig_sample = sampler.sample
    sampler.sample = make_mrv2_sample_wrapper(
        sampler=sampler,
        runtime=runtime,
        runner=runner,
        orig_sample=orig_sample,
    )
    setattr(sampler, _MARK_ATTR, True)
