#!/usr/bin/env python3
"""Bridge for optional same-step sampler intervention."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from types import SimpleNamespace

import torch

from tllm.runtime.sampler_bridge.dense_backend import modify_rows_dense
from tllm.runtime.sampler_bridge import exact_backend
from tllm.runtime.instrumentation import maybe_record_function
from tllm.runtime.sampler_bridge.minp_kernel import MinPCandidateKernelRequest, select_minp_candidate_kernel
from tllm.runtime.sampler_bridge.provider import (
    SamplerModifierProvider,
    validate_prepared_intervention,
)
from tllm.runtime.sampler_bridge.types import (
    CandidateModifierState,
    CandidateSampleResult,
    SamplerStepView,
)

_SAMPLING_EPS = 1e-5
import os
_ENABLE_EXACT_CANDIDATE_SAMPLING = os.getenv("TLLM_ENABLE_EXACT_CANDIDATE_SAMPLING", "") == "1"
_ENABLE_TRITON_MINP_KEEP_MASK = os.getenv("TLLM_ENABLE_TRITON_MINP_KEEP_MASK", "") == "1"


def _unwrap_sampled(result: Any) -> torch.Tensor:
    """Accept both legacy tensor returns and vLLM >= 0.25 tuple returns."""
    if isinstance(result, tuple):
        return result[0]
    return result


def _sample_rows_vanilla(
    *,
    sampler: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    row_ids: torch.Tensor,
) -> torch.Tensor:
    subset_logits = logits.index_select(0, row_ids)
    top_k = getattr(sampling_metadata, "top_k", None)
    if top_k is not None:
        top_k = top_k.index_select(0, row_ids.to(device=top_k.device, dtype=torch.long))
    top_p = getattr(sampling_metadata, "top_p", None)
    if top_p is not None:
        top_p = top_p.index_select(0, row_ids.to(device=top_p.device, dtype=torch.long))
    generators: dict[int, torch.Generator] = {}
    source_generators = getattr(sampling_metadata, "generators", {})
    if source_generators:
        for new_i, row_i in enumerate(row_ids.cpu().tolist()):
            generator = source_generators.get(int(row_i))
            if generator is not None:
                generators[int(new_i)] = generator
    if subset_logits.device.type != "cuda":
        return _unwrap_sampled(sampler.topk_topp_sampler.forward_native(subset_logits, generators, top_k, top_p)).to(dtype=torch.long)
    return _unwrap_sampled(sampler.topk_topp_sampler(subset_logits, generators, top_k, top_p)).to(dtype=torch.long)


def _sample_full_vanilla(
    *,
    sampler: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
) -> torch.Tensor:
    if logits.device.type != "cuda":
        return _unwrap_sampled(sampler.topk_topp_sampler.forward_native(
            logits,
            getattr(sampling_metadata, "generators", {}),
            getattr(sampling_metadata, "top_k", None),
            getattr(sampling_metadata, "top_p", None),
        )).to(dtype=torch.long)
    return _unwrap_sampled(sampler.topk_topp_sampler(
        logits,
        getattr(sampling_metadata, "generators", {}),
        getattr(sampling_metadata, "top_k", None),
        getattr(sampling_metadata, "top_p", None),
    )).to(dtype=torch.long)


def _sample_rows_dense(
    *,
    sampler: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    state: Any,
    row_ids: torch.Tensor,
    greedy: bool,
) -> torch.Tensor:
    modified = modify_rows_dense(logits=logits, state=state, row_ids=row_ids)
    if greedy:
        return modified.argmax(dim=-1).to(dtype=torch.long)
    top_k = getattr(sampling_metadata, "top_k", None)
    if top_k is not None:
        top_k = top_k.index_select(0, row_ids.to(device=top_k.device, dtype=torch.long))
    top_p = getattr(sampling_metadata, "top_p", None)
    if top_p is not None:
        top_p = top_p.index_select(0, row_ids.to(device=top_p.device, dtype=torch.long))
    source_generators = getattr(sampling_metadata, "generators", {})
    generators = {}
    if source_generators:
        generators = {
            int(new_i): generator
            for new_i, row_i in enumerate(row_ids.cpu().tolist())
            if (generator := source_generators.get(int(row_i))) is not None
        }
    return _sample_full_vanilla(
        sampler=sampler,
        logits=modified,
        sampling_metadata=SimpleNamespace(
            top_k=top_k,
            top_p=top_p,
            generators=generators,
        ),
    )


def _sample_rows_exact_via_sampler(
    *,
    sampler: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    state: Any,
    row_ids: torch.Tensor,
    greedy: bool,
    all_rows: bool = False,
) -> torch.Tensor:
    modified, generators = exact_backend.build_modified_logits_exact(
        logits=logits,
        sampling_metadata=sampling_metadata,
        state=state,
        row_ids=row_ids,
        greedy=greedy,
        all_rows=all_rows,
    )
    if greedy:
        return modified.argmax(dim=-1).to(dtype=torch.long)
    return _sample_full_vanilla(
        sampler=sampler,
        logits=modified,
        sampling_metadata=SimpleNamespace(
            top_k=None,
            top_p=None,
            generators=generators,
        ),
    )


def _sample_rows_exact_candidates(
    *,
    logits: torch.Tensor,
    sampling_metadata: Any,
    state: Any,
    row_ids: torch.Tensor,
    all_rows: bool = False,
) -> torch.Tensor | None:
    if (
        all_rows
        and getattr(sampling_metadata, "top_k", None) is None
        and getattr(sampling_metadata, "top_p", None) is None
        and getattr(sampling_metadata, "min_p", None) is not None
        and not bool(getattr(sampling_metadata, "generators", {}) or {})
    ):
        kernel = select_minp_candidate_kernel(
            prefer_triton=_ENABLE_TRITON_MINP_KEEP_MASK,
            logits_device=logits.device,
            logits=logits,
        )
        result = kernel.sample(
            MinPCandidateKernelRequest(
                logits=logits,
                min_p=getattr(sampling_metadata, "min_p"),
                state=state,
                greedy=False,
            )
        )
        _record_candidate_kernel_stats(result)
        return result.sampled_token_ids
    return exact_backend.sample_rows_exact_candidates(
        logits=logits,
        sampling_metadata=sampling_metadata,
        state=state,
        row_ids=row_ids,
        all_rows=all_rows,
    )


def _record_candidate_kernel_stats(result: CandidateSampleResult) -> None:
    stats = dict(result.debug_stats or {})
    try:
        from tllm.runtime import residual_runtime

        runtime = residual_runtime.RUNTIME
    except Exception:
        return
    precompute = runtime.sampler_precompute
    candidate_count = int(stats.get("candidate_count", 0) or 0)
    row_count = int(stats.get("row_count", 0) or 0)
    precompute.candidate_sample_count += 1
    precompute.candidate_token_count += candidate_count
    precompute.candidate_row_count += row_count
    precompute.candidate_max_count = max(precompute.candidate_max_count, candidate_count)
    kernel_name = str(stats.get("kernel", "") or "")
    if kernel_name.startswith("triton"):
        precompute.candidate_kernel_triton_count += 1
    elif kernel_name:
        precompute.candidate_kernel_torch_count += 1
    if "fallback_reason" in stats:
        precompute.candidate_kernel_fallback_count += 1


def _classify_temperature_mode(sampling_metadata: Any) -> str:
    if bool(getattr(sampling_metadata, "all_greedy", False)):
        return "all_greedy"
    if bool(getattr(sampling_metadata, "all_random", False)):
        return "all_random"
    temperature = getattr(sampling_metadata, "temperature", None)
    if temperature is None:
        return "all_random"
    if not isinstance(temperature, torch.Tensor):
        return "all_greedy" if float(temperature) < _SAMPLING_EPS else "all_random"
    if temperature.numel() <= 0:
        return "all_random"
    if temperature.device.type == "cuda":
        return "mixed"
    if bool(torch.all(temperature < _SAMPLING_EPS)):
        return "all_greedy"
    if bool(torch.all(temperature >= _SAMPLING_EPS)):
        return "all_random"
    return "mixed"


def _build_dense_modified_logits(
    *,
    logits: torch.Tensor,
    state: Any,
    affected_rows: torch.Tensor,
    in_place: bool = False,
) -> torch.Tensor:
    if int(affected_rows.numel()) == 0:
        return logits
    if int(affected_rows.numel()) == int(logits.shape[0]) and state.precomputed_dense_logits is not None:
        dense = state.precomputed_dense_logits.to(device=logits.device, dtype=logits.dtype)
        if in_place:
            logits.mul_(1.0 + float(state.beta))
            logits.add_(dense, alpha=-float(state.beta))
            return logits
        return ((1.0 + float(state.beta)) * logits) - (float(state.beta) * dense)
    modified = logits if in_place else logits.clone()
    modified_rows = modify_rows_dense(logits=logits, state=state, row_ids=affected_rows)
    modified.index_copy_(0, affected_rows, modified_rows)
    return modified


def apply_pre_filter_dense_logits(
    *, logits: torch.Tensor, state: CandidateModifierState
) -> torch.Tensor:
    """Apply a dense modifier to raw logits before sampler processors."""

    affected_rows = state.affected_row_ids.to(
        device=logits.device,
        dtype=torch.long,
    )
    return _build_dense_modified_logits(
        logits=logits,
        state=state,
        affected_rows=affected_rows,
        in_place=False,
    )


def sample_with_optional_modifier(
    *,
    sampler: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    view: SamplerStepView | None,
    provider: SamplerModifierProvider | None,
    greedy_sampled: torch.Tensor | None,
    before_draw: Callable[[], None] | None = None,
    prepared_state: CandidateModifierState | None = None,
) -> torch.Tensor | None:
    with maybe_record_function("distiller.sample_with_optional_modifier"):
        if view is None or provider is None or not provider.is_active():
            return None
        # The parameter-read fence must publish *before* the provider reads
        # any Region output: dispatch joins side-stream work onto the main
        # stream, and provider consumption after the join observes complete
        # results.  The reverse order raced side Regions signed before
        # sampler.pre_draw.
        if prepared_state is None:
            if before_draw is not None:
                before_draw()
            state = provider.prepare_step(view)
        else:
            state = prepared_state
        if state is None or int(state.affected_row_ids.numel()) <= 0:
            return None
        validate_prepared_intervention(provider, state)
        try:
            runtime = view.runner._tllm_runtime
        except AttributeError:
            runtime = None
        if runtime is not None:
            runtime.sampler_precompute.intervention_state_count += 1
        affected_rows = state.affected_row_ids.to(device=logits.device, dtype=torch.long)
        rows = int(logits.shape[0])
        temperature_mode = _classify_temperature_mode(sampling_metadata)
        if greedy_sampled is not None and temperature_mode == "mixed":
            greedy_mask = torch.zeros((rows,), device=logits.device, dtype=torch.bool)
            temp = getattr(sampling_metadata, "temperature", None)
            if temp is not None:
                greedy_mask = temp.to(device=logits.device) < _SAMPLING_EPS
            greedy_rows = greedy_mask.nonzero(as_tuple=False).view(-1)
        else:
            greedy_mask = None
            greedy_rows = affected_rows.new_empty((0,), dtype=torch.long)

        from tllm import core as _core

        plan = _core.plan_sampler_step(
            state.backend,
            temperature_mode,
            rows,
            int(affected_rows.numel()),
            int(greedy_rows.numel()),
            greedy_sampled is not None,
            _ENABLE_EXACT_CANDIDATE_SAMPLING,
        )

        if plan.path in ("full_dense_greedy", "full_dense_random"):
            with maybe_record_function("distiller.sample_with_optional_modifier.full_dense"):
                modified = _build_dense_modified_logits(
                    logits=logits,
                    state=state,
                    affected_rows=affected_rows,
                    in_place=True,
                )
                if plan.path == "full_dense_greedy":
                    return modified.argmax(dim=-1).to(dtype=torch.long)
                return _sample_full_vanilla(
                    sampler=sampler,
                    logits=modified,
                    sampling_metadata=sampling_metadata,
                )

        if plan.path in ("full_exact_greedy", "full_exact_random"):
            with maybe_record_function("distiller.sample_with_optional_modifier.full_exact"):
                if plan.path == "full_exact_greedy":
                    return _sample_rows_exact_via_sampler(
                        sampler=sampler,
                        logits=logits,
                        sampling_metadata=sampling_metadata,
                        state=state,
                        row_ids=affected_rows,
                        greedy=True,
                        all_rows=True,
                    )
                if plan.use_candidate_kernel:
                    candidate_sampled = _sample_rows_exact_candidates(
                        logits=logits,
                        sampling_metadata=sampling_metadata,
                        state=state,
                        row_ids=affected_rows,
                        all_rows=True,
                    )
                    if candidate_sampled is not None:
                        return candidate_sampled
                return _sample_rows_exact_via_sampler(
                    sampler=sampler,
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                    state=state,
                    row_ids=affected_rows,
                    greedy=False,
                    all_rows=True,
                )

        # Partitioned path. The Rust core owns the branch decision; row
        # values stay on-device (mask/nonzero partitioning is executor work).
        if plan.dense_inplace_applied:
            # LEGACY quirk (pinned by the oracle): mixed mode after the
            # full-row dense branch already modified logits in place.
            _build_dense_modified_logits(
                logits=logits,
                state=state,
                affected_rows=affected_rows,
                in_place=True,
            )

        def _sample_affected(row_ids: torch.Tensor, *, greedy: bool) -> torch.Tensor:
            if state.backend == "pre_filter_dense":
                return _sample_rows_dense(
                    sampler=sampler,
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                    state=state,
                    row_ids=row_ids,
                    greedy=greedy,
                )
            return _sample_rows_exact_via_sampler(
                sampler=sampler,
                logits=logits,
                sampling_metadata=sampling_metadata,
                state=state,
                row_ids=row_ids,
                greedy=greedy,
                all_rows=False,
            )

        if plan.base != "empty":
            # all_greedy partitioned: only the affected (greedy) rows resample.
            if plan.base == "greedy_sampled":
                final = greedy_sampled.to(device=logits.device, dtype=torch.long).clone()
            else:
                final = logits.argmax(dim=-1).to(dtype=torch.long)
            final[affected_rows] = _sample_affected(affected_rows, greedy=True)
            return final

        final = torch.empty((rows,), device=logits.device, dtype=torch.long)
        affected_mask = torch.zeros((rows,), device=logits.device, dtype=torch.bool)
        affected_mask[affected_rows] = True

        if greedy_mask is None:
            unaffected_greedy_rows = affected_rows.new_empty((0,), dtype=torch.long)
        else:
            unaffected_greedy = (~affected_mask) & greedy_mask
            unaffected_greedy_rows = unaffected_greedy.nonzero(as_tuple=False).view(-1)
        if unaffected_greedy_rows.numel() > 0 and greedy_sampled is not None:
            final[unaffected_greedy] = greedy_sampled.to(device=final.device, dtype=torch.long)[unaffected_greedy]

        if greedy_mask is None:
            affected_greedy_rows = affected_rows.new_empty((0,), dtype=torch.long)
        else:
            affected_greedy = affected_mask & greedy_mask
            affected_greedy_rows = affected_greedy.nonzero(as_tuple=False).view(-1)
        if affected_greedy_rows.numel() > 0:
            final[affected_greedy_rows] = _sample_affected(affected_greedy_rows, greedy=True)

        if greedy_mask is None:
            unaffected_random_rows = (~affected_mask).nonzero(as_tuple=False).view(-1)
        else:
            unaffected_random = (~affected_mask) & (~greedy_mask)
            unaffected_random_rows = unaffected_random.nonzero(as_tuple=False).view(-1)
        if unaffected_random_rows.numel() > 0:
            final[unaffected_random_rows] = _sample_rows_vanilla(
                sampler=sampler,
                logits=logits,
                sampling_metadata=sampling_metadata,
                row_ids=unaffected_random_rows,
            )

        if greedy_mask is None:
            affected_random_rows = affected_rows
        else:
            affected_random = affected_mask & (~greedy_mask)
            affected_random_rows = affected_random.nonzero(as_tuple=False).view(-1)
        if affected_random_rows.numel() > 0:
            final[affected_random_rows] = _sample_affected(affected_random_rows, greedy=False)

        return final
