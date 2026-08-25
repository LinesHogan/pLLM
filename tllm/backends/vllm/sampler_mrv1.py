#!/usr/bin/env python3
"""Sampler interceptor for vLLM 0.25+ MRv1 (``vllm.v1.sample.sampler.Sampler``).

0.25+ changed the sampler protocol in three ways versus the legacy patch in
``tllm.runtime.vllm_patch.sampler_patch``:

* ``Sampler.sample()`` returns ``(sampled, processed_logprobs)`` and accepts
  ``logprobs_mode_override``.
* ``apply_temperature`` takes an ``all_random`` guard and always divides
  in-place.
* min-p is a native argmax-invariant logits processor, so tLLM no longer
  patches ``InputBatch``/``SamplingMetadata``; the processor runs inside the
  wrapped ``sample`` before the bridge sees the logits, which keeps the
  declared "filter first, intervene on survivors" semantics intact (min-p,
  top-k and top-p masking commute).

The wrapper mirrors the upstream ``Sampler.sample`` control flow exactly so
that an inactive provider is bit-for-bit the upstream algorithm, and injects
the modifier between the argmax-invariant processors and the top-k/top-p
draw.
"""

from __future__ import annotations

from typing import Any

import torch

from tllm.runtime.vllm_patch.sampler_patch import (
    dispatch_sampler_pre_draw,
    _notify_after_sample,
    _resolve_provider,
)

_SAMPLING_EPS = 1e-5


def _processed_logprobs_for_mode(
    *,
    sampler: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    logprobs_mode: Any,
) -> torch.Tensor | None:
    """Mirror upstream processed-logits/logprobs handling for modified rows.

    When the intervention modified only a subset of rows (exact backends)
    the full modified logits matrix is not materialised; in that case the
    post-temperature pre-intervention logits are used. Raw logprobs modes
    are computed by ``Sampler.forward`` from the pre-sample logits and are
    unaffected here.
    """
    if sampling_metadata.max_num_logprobs is None and not sampling_metadata.logprob_token_ids:
        return None
    if logprobs_mode == "processed_logits":
        return logits
    if logprobs_mode == "processed_logprobs":
        return sampler.compute_logprobs(logits)
    return None


def _standard_completion(
    *,
    sampler: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    greedy_sampled: torch.Tensor | None,
    logprobs_mode: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Finish the upstream sampling algorithm after temperature/processors.

    Identical to the tail of upstream ``Sampler.sample``; used when the
    provider produced no candidate state so parity holds bit-for-bit.
    """
    if sampling_metadata.all_greedy:
        processed = _processed_logprobs_for_mode(
            sampler=sampler,
            logits=logits,
            sampling_metadata=sampling_metadata,
            logprobs_mode=logprobs_mode,
        )
        return greedy_sampled, processed

    random_sampled, processed = sampler.topk_topp_sampler(
        logits,
        sampling_metadata.generators,
        sampling_metadata.top_k,
        sampling_metadata.top_p,
    )
    if processed is None:
        processed = _processed_logprobs_for_mode(
            sampler=sampler,
            logits=logits,
            sampling_metadata=sampling_metadata,
            logprobs_mode=logprobs_mode,
        )
    if greedy_sampled is None:
        return random_sampled, processed
    sampled = torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,
        greedy_sampled,
        random_sampled,
        out=greedy_sampled,
    )
    return sampled, processed


def wrapped_sampler_sample_v027(
    *,
    sampler: Any,
    logits: torch.Tensor,
    sampling_metadata: Any,
    logprobs_mode_override: Any = None,
    orig_sample: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from tllm.runtime.sampler_bridge.bridge import sample_with_optional_modifier
    from tllm.runtime.sampler_bridge.runtime_view import build_sampler_step_view

    runtime = getattr(sampler, "_tllm_runtime", None)
    runner = getattr(sampler, "_tllm_runner", None)
    provider = _resolve_provider(runtime)
    if runtime is None or runner is None or provider is None or not provider.is_active():
        return orig_sample(sampler, logits, sampling_metadata, logprobs_mode_override=logprobs_mode_override)

    logprobs_mode = logprobs_mode_override or sampler.logprobs_mode
    all_random = bool(sampling_metadata.all_random)
    greedy_sampled = None if all_random else sampler.greedy_sample(logits)

    # Mirror upstream ordering: temperature first (skipped for all-greedy),
    # then argmax-invariant processors (min-p lives here natively in 0.25+).
    if not sampling_metadata.all_greedy:
        logits = sampler.apply_temperature(logits, sampling_metadata.temperature, all_random)
        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

    modified_logits = logits
    view = build_sampler_step_view(
        runtime=runtime,
        runner=runner,
        model=getattr(runner, "model", None),
        logits=logits,
        sampling_metadata=sampling_metadata,
    )
    sampled = sample_with_optional_modifier(
        sampler=sampler,
        logits=logits,
        sampling_metadata=sampling_metadata,
        view=view,
        provider=provider,
        greedy_sampled=greedy_sampled,
        before_draw=lambda: dispatch_sampler_pre_draw(
            runtime=runtime,
            runner=runner,
            device=logits.device,
        ),
    )

    if sampled is None:
        # No candidate state: finish with the unmodified upstream algorithm.
        return _standard_completion(
            sampler=sampler,
            logits=logits,
            sampling_metadata=sampling_metadata,
            greedy_sampled=greedy_sampled,
            logprobs_mode=logprobs_mode,
        )

    sampled = sampled.to(dtype=torch.long)
    if view is not None:
        _notify_after_sample(provider=provider, view=view, sampled_token_ids=sampled)
    processed = _processed_logprobs_for_mode(
        sampler=sampler,
        logits=modified_logits,
        sampling_metadata=sampling_metadata,
        logprobs_mode=logprobs_mode,
    )
    return sampled, processed


def install_sampler_patch_v027(*, core: Any) -> None:
    """Patch ``Sampler.sample`` once; instance routing via ``_tllm_runtime``."""
    if getattr(core, "_V027_SAMPLER_PATCH_INSTALLED", False):
        return
    from vllm.v1.sample.sampler import Sampler

    orig_sample = Sampler.sample

    def _wrapped(
        self: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        logprobs_mode_override: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return wrapped_sampler_sample_v027(
            sampler=self,
            logits=logits,
            sampling_metadata=sampling_metadata,
            logprobs_mode_override=logprobs_mode_override,
            orig_sample=orig_sample,
        )

    Sampler.sample = _wrapped
    core._ORIG_VLLM_SAMPLER_SAMPLE = orig_sample
    core._V027_SAMPLER_PATCH_INSTALLED = True
