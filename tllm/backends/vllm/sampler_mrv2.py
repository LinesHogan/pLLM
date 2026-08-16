#!/usr/bin/env python3
"""Instance-scoped sampler interceptor for vLLM 0.25+ MRv2.

The MRv2 sampler (``vllm.v1.worker.gpu.sample.sampler.Sampler``) is a plain
callable. ``sample()`` applies bias/penalties/bad-words/temperature/min-p
via ``apply_sampling_params(skip_top_k_top_p=True)``, then filters with
top-k/top-p (either fused inside FlashInfer or via ``apply_top_k_top_p``)
and draws with ``gumbel_sample``.

ESamp requires the "filter first, intervene on survivors" stage, which does
not exist inside the fused FlashInfer kernel. The interceptor therefore:

* leaves the instance untouched when no provider is active (bit-for-bit the
  upstream path, FlashInfer included);
* when a provider is active, forces the non-FlashInfer path for that step,
  applies top-k/top-p explicitly, reweights only the surviving candidates
  with ``(1+beta)*logit - beta*distiller_logit``, and draws with the same
  ``gumbel_sample`` call (same seeds/positions, so RNG consumption matches
  the unmodified non-FlashInfer path).

The class is never patched globally; the wrapper is installed on the
sampler instance owned by the bound runner.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from tllm.runtime.sampler_bridge.types import CandidateModifierState, SamplerStepView
from tllm.runtime.vllm_patch.sampler_patch import _notify_after_sample, _resolve_provider

_MARK_ATTR = "_tllm_mrv2_sampler_wrapped"


def _build_mrv2_step_view(*, runtime: Any, runner: Any, logits: torch.Tensor) -> SamplerStepView | None:
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
        prompt_idxs = ()
    else:
        prompt_idx_tensor = None
        prompt_idxs = tuple(int(x) for x in list(getattr(runtime, "decode_prompt_idxs", []))[:decode_count])
        if len(prompt_idxs) != decode_count:
            return None
    sample_idx_tensor = getattr(runtime, "decode_sample_idx_tensor", None)
    if isinstance(sample_idx_tensor, torch.Tensor) and int(sample_idx_tensor.numel()) >= decode_count:
        sample_idx_tensor = sample_idx_tensor[:decode_count]
        sample_idxs = ()
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

        view = _build_mrv2_step_view(runtime=runtime, runner=runner, logits=logits)
        state = provider.prepare_step(view) if view is not None else None
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

        from vllm.config.model import PROCESSED_LOGPROBS_MODES
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

        processed_logits = sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            skip_top_k_top_p=True,
        )
        top_k, top_p = sampler.sampling_states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)

        if state.backend == "pre_filter_dense":
            dense = state.precomputed_dense_logits
            if dense is None:
                pred_hidden = state.pred_hidden
                dense = F.linear(
                    pred_hidden.to(dtype=state.lm_head_weight.dtype),
                    state.lm_head_weight,
                    state.lm_head_bias,
                ).to(torch.float32)
            dense = dense.to(device=processed_logits.device, dtype=processed_logits.dtype)
            affected = state.affected_row_ids.to(device=processed_logits.device, dtype=torch.long)
            sub = processed_logits[affected]
            sub.mul_(1.0 + float(state.beta))
            sub.add_(dense[: int(affected.numel())], alpha=-float(state.beta))
            processed_logits[affected] = sub
            filtered = apply_top_k_top_p(processed_logits, top_k, top_p)
        else:
            # post-filter: filter the base logits first, then reweight only
            # the surviving candidates. FlashInfer fuses filter+draw and has
            # no hook point between them, so this step always takes the
            # explicit path.
            filtered = apply_top_k_top_p(processed_logits, top_k, top_p)
            filtered = _apply_intervention(filtered_logits=filtered, state=state)

        sampled = gumbel_sample(
            filtered,
            expanded_idx_mapping,
            sampler.sampling_states.temperature.gpu,
            sampler.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
            use_fp64=sampler.use_fp64_gumbel,
        )
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
