#!/usr/bin/env python3
"""Thin sampler patch that routes through the tLLM sampler bridge."""

from __future__ import annotations

from typing import Any

import torch

from tllm.backends.protocol import BackendObservableEvent
from tllm.runtime.instrumentation import maybe_record_function
from tllm.runtime.sampler_bridge.min_p import apply_min_p

_PATCH_INSTALLED = False
_ORIG_VLLM_SAMPLER_SAMPLE = None
_SAMPLING_EPS = 1e-5
def _resolve_provider(runtime: Any) -> Any:
    return None


def sample_with_optional_modifier(*args, **kwargs):
    from tllm.runtime.sampler_bridge.bridge import sample_with_optional_modifier as _impl

    return _impl(*args, **kwargs)


def build_sampler_step_view(*args, **kwargs):
    from tllm.runtime.sampler_bridge.runtime_view import build_sampler_step_view as _impl

    return _impl(*args, **kwargs)


def _notify_after_sample(*, provider: Any, view: Any, sampled_token_ids: torch.Tensor) -> None:
    after_sample = getattr(provider, "after_sample", None)
    if callable(after_sample):
        after_sample(view, sampled_token_ids)


def bind_runner_sampler(*, runtime: Any, runner: Any) -> None:
    sampler = getattr(runner, "sampler", None)
    if sampler is None:
        return
    setattr(sampler, "_tllm_runtime", runtime)
    setattr(sampler, "_tllm_runner", runner)
    setattr(runner, "_tllm_runtime", runtime)


def ensure_sampler_precompute_buffers(*, runtime: Any, runner: Any) -> None:
    provider = _resolve_provider(runtime)
    if provider is None or not provider.is_active():
        return
    ensure = getattr(provider, "ensure_runtime_buffers", None)
    if callable(ensure):
        ensure(runtime=runtime, runner=runner)


def maybe_prepare_sampler_decode_step(*, runtime: Any, runner: Any) -> None:
    provider = _resolve_provider(runtime)
    if provider is None or not provider.is_active():
        precompute = runtime.sampler_precompute
        precompute.reset_decode_step(int(getattr(runtime, "event_step_id", -1)))
        precompute.port_enabled = False
        return
    runtime.sampler_precompute.port_enabled = True
    prepare = getattr(provider, "maybe_prepare_decode_step", None)
    if callable(prepare):
        prepare(runtime=runtime, runner=runner)


def runtime_has_active_sampler_provider(runtime: Any) -> bool:
    provider = _resolve_provider(runtime)
    return bool(provider is not None and provider.is_active())


def dispatch_sampler_pre_draw(
    *,
    runtime: Any,
) -> None:
    """Advance the signed sampler pre-draw event."""
    from tllm.runtime.vllm_patch import common_hooks as _common_hooks

    _common_hooks.dispatch_runtime_event(
        runtime=runtime,
        event=BackendObservableEvent.SAMPLER_PRE_DRAW,
    )


def maybe_capture_source_precompute(*, runtime: Any, runner: Any, layer_path: str) -> None:
    provider = _resolve_provider(runtime)
    if provider is None or not provider.is_active():
        return
    capture = getattr(provider, "maybe_capture_source_precompute", None)
    if callable(capture):
        capture(runtime=runtime, runner=runner, layer_path=layer_path)


def _vanilla_sample(*, sampler: Any, logits: torch.Tensor, sampling_metadata: Any) -> torch.Tensor:
    assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
    if sampling_metadata.all_random:
        greedy_sampled = None
    else:
        greedy_sampled = sampler.greedy_sample(logits)
        if sampling_metadata.all_greedy:
            return greedy_sampled

    assert sampling_metadata.temperature is not None
    logits = sampler.apply_temperature(logits, sampling_metadata.temperature)
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits = processor.apply(logits)

    logits = apply_min_p(logits, getattr(sampling_metadata, "min_p", None))

    random_sampled = sampler.topk_topp_sampler(
        logits,
        sampling_metadata.generators,
        sampling_metadata.top_k,
        sampling_metadata.top_p,
    )
    if greedy_sampled is None:
        return random_sampled.to(dtype=torch.long)
    return torch.where(
        sampling_metadata.temperature < _SAMPLING_EPS,
        greedy_sampled,
        random_sampled,
        out=greedy_sampled,
    ).to(dtype=torch.long)


def wrapped_sampler_sample(*, sampler: Any, logits: torch.Tensor, sampling_metadata: Any) -> torch.Tensor:
    runtime = getattr(sampler, "_tllm_runtime", None)
    runner = getattr(sampler, "_tllm_runner", None)
    provider = _resolve_provider(runtime)
    if runtime is None or runner is None or provider is None or not provider.is_active():
        if getattr(sampling_metadata, "min_p", None) is not None:
            return _vanilla_sample(sampler=sampler, logits=logits, sampling_metadata=sampling_metadata)
        if _ORIG_VLLM_SAMPLER_SAMPLE is not None:
            return _ORIG_VLLM_SAMPLER_SAMPLE(sampler, logits, sampling_metadata)
        return _vanilla_sample(sampler=sampler, logits=logits, sampling_metadata=sampling_metadata)

    assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
    if sampling_metadata.all_random:
        greedy_sampled = None
    else:
        greedy_sampled = sampler.greedy_sample(logits)

    temperature = getattr(sampling_metadata, "temperature", None)
    logits_for_sampling = logits if temperature is None else sampler.apply_temperature(logits, temperature)
    for processor in sampling_metadata.logitsprocs.argmax_invariant:
        logits_for_sampling = processor.apply(logits_for_sampling)

    view = build_sampler_step_view(
        runtime=runtime,
        runner=runner,
        model=getattr(runner, "model", None),
        logits=logits_for_sampling,
        sampling_metadata=sampling_metadata,
    )
    sampled = sample_with_optional_modifier(
        sampler=sampler,
        logits=logits_for_sampling,
        sampling_metadata=sampling_metadata,
        view=view,
        provider=provider,
        greedy_sampled=greedy_sampled,
        before_draw=lambda: dispatch_sampler_pre_draw(
            runtime=runtime,
        ),
    )
    if sampled is not None:
        sampled = sampled.to(dtype=torch.long)
        if view is not None:
            _notify_after_sample(provider=provider, view=view, sampled_token_ids=sampled)
        return sampled
    if _ORIG_VLLM_SAMPLER_SAMPLE is not None:
        return _ORIG_VLLM_SAMPLER_SAMPLE(sampler, logits, sampling_metadata)
    return _vanilla_sample(sampler=sampler, logits=logits, sampling_metadata=sampling_metadata)


def maybe_schedule_sampler_precompute(*, runtime: Any, runner: Any, layer_path: str) -> None:
    if str(layer_path).strip() != str(getattr(runtime, "source_resolved_path", "") or "").strip():
        return
    provider = _resolve_provider(runtime)
    if provider is None or not provider.is_active():
        return
    schedule = getattr(provider, "maybe_schedule_precompute", None)
    if callable(schedule):
        schedule(runtime=runtime, runner=runner)


def install_sampler_patch(*, core: Any) -> None:
    global _PATCH_INSTALLED, _ORIG_VLLM_SAMPLER_SAMPLE
    if _PATCH_INSTALLED:
        return
    from vllm.v1.sample.sampler import Sampler

    _ORIG_VLLM_SAMPLER_SAMPLE = Sampler.sample

    def _wrapped(self: Any, logits: torch.Tensor, sampling_metadata: Any) -> torch.Tensor:
        return wrapped_sampler_sample(sampler=self, logits=logits, sampling_metadata=sampling_metadata)

    Sampler.sample = _wrapped
    core._ORIG_VLLM_SAMPLER_SAMPLE = _ORIG_VLLM_SAMPLER_SAMPLE
    _PATCH_INSTALLED = True
