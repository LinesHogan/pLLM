#!/usr/bin/env python3
"""Residual-stream runtime host: public API and hook-facing "core" module.

This module is intentionally thin. The implementation lives in focused
modules:

* ``tllm.runtime.state`` — ``ResidualRuntimeConfig`` / ``ResidualRuntimeState``
  and the shared ``RUNTIME`` object;
* ``tllm.runtime.resources`` — sampler precompute buffers, caches, streams;
* ``tllm.runtime.stats`` — statistics containers and collectors;
* ``tllm.runtime.lifecycle`` — runner patch installation and ``make_llm``;
* ``tllm.runtime.vllm_patch.port_runtime_hooks`` — the hook bodies, driven
  with this module as the ``core`` handle.

The module-level ``GPUModelRunner`` / ``_ORIG_*`` / ``_PATCH_INSTALLED``
globals stay here because backends and hook modules address them through the
``core`` module object.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence

import torch

from tllm.common.state import ensure_v1_env, pick_common_attn_metadata, resolve_object_by_path
from tllm.consumers.base import BaseConsumer
from tllm.contracts.consumer_stats import TrainingStats

# Backward-compatible alias: ESamp's stats name, canonical type is TrainingStats.
ESampStats = TrainingStats
from tllm.contracts.runtime_context import RunnerLike
from tllm.producer.decode import compute_decode_localization
from tllm.runtime import lifecycle as _lifecycle
from tllm.runtime import stats as _stats
from tllm.runtime.config import normalize_adaptation_stream_mode, normalize_model_bank_forward_backend
from tllm.runtime.consumer_registry import ConsumerRegistry
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.runtime.consumer_compat import synchronize as synchronize_consumer
from tllm.runtime.resources import (
    SamplerPrecomputeBuffers,
    SamplerPrecomputeCache,
    SamplerPrecomputeState,
)
from tllm.runtime.sampler_bridge.types import SamplerBackend, normalize_sampler_backend
from tllm.runtime.state import RUNTIME, ResidualRuntimeConfig, ResidualRuntimeState, RuntimeConsumer
from tllm.runtime.stats import DistillerTimingStats, GraphDebugStats, PathHotspotStats
from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks

ensure_v1_env()

GPUModelRunner = None
_ORIG_LOAD_MODEL = None
_ORIG_PREPARE_INPUTS = None
_ORIG_EXECUTE_MODEL = None
_PATCH_INSTALLED = False

MODEL_HOOK_FLAG = "esamp_hook_installed"
MODEL_HOOK_SPEC_ATTR = "esamp_hook_spec"


def _ensure_runner_binding() -> None:
    _lifecycle.ensure_runner_binding(sys.modules[__name__])


def _ensure_dispatch_registry() -> ConsumerRegistry:
    if RUNTIME.consumer_registry is None:
        RUNTIME.consumer_registry = ConsumerRegistry()
    return RUNTIME.consumer_registry


def _refresh_dispatch_plan() -> None:
    registry = _ensure_dispatch_registry()
    consumers = registry.consumers()
    if isinstance(RUNTIME.consumer, BaseConsumer):
        consumers = consumers + [RUNTIME.consumer]
    if not consumers:
        RUNTIME.dispatch_plan = None
        RUNTIME.execution_plan = None
        return
    plan = DispatchPlan.build(consumers)
    RUNTIME.dispatch_plan = None if plan.is_empty() else plan
    # Fail fast at startup when a consumer reads a migrated port kind that no
    # registered producer can serve (kinds still served by legacy hardcoded
    # producers are not enforced yet).
    from tllm.runtime.execution_plan import compile_default_plan

    RUNTIME.execution_plan = compile_default_plan(consumers=consumers)


def register_dispatch_consumer(consumer: BaseConsumer) -> None:
    _ensure_dispatch_registry().register(consumer)
    _refresh_dispatch_plan()


def replace_dispatch_consumers(consumers: Sequence[BaseConsumer]) -> None:
    _ensure_dispatch_registry().replace_all(consumers)
    _refresh_dispatch_plan()


def clear_dispatch_consumers() -> None:
    _ensure_dispatch_registry().clear()
    _refresh_dispatch_plan()


def set_runtime_consumer(consumer: BaseConsumer | None) -> None:
    RUNTIME.consumer = consumer
    _refresh_dispatch_plan()


def runner_uses_compilation_or_cudagraph(runner: RunnerLike) -> bool:
    """Legacy predicate kept for call sites; semantics now live in
    ``tllm.runtime.graph_policy.resolve_graph_mode`` (PIECEWISE/FULL ->
    True)."""
    from tllm.runtime import graph_policy as _graph_policy

    mode = _graph_policy.resolve_graph_mode(runner)
    if mode != _graph_policy.GraphMode.EAGER:
        return True

    # Structural fallback for runners that only expose the legacy flags.
    use_cuda_graph = getattr(runner, "use_cuda_graph", None)
    if use_cuda_graph is True:
        return True

    cfg = getattr(runner, "compilation_config", None)
    if cfg is None:
        return False

    level_raw = getattr(cfg, "level", 0)
    try:
        level = int(level_raw)
    except Exception:
        level = 0
    return level > 0


def infer_hidden_dtype(layer: torch.nn.Module) -> torch.dtype:
    for p in layer.parameters():
        return p.dtype
    for b in layer.buffers():
        return b.dtype
    return torch.float32


def resolve_module_by_path_with_fallback(model: torch.nn.Module, path: str) -> tuple[torch.nn.Module, str]:
    from tllm.common.path_resolution import candidate_capture_paths

    errors: list[str] = []
    for candidate in candidate_capture_paths(path):
        try:
            obj = resolve_object_by_path(model, candidate)
        except RuntimeError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if not isinstance(obj, torch.nn.Module):
            errors.append(f"{candidate}: resolved to non-module {type(obj).__name__}")
            continue
        return obj, candidate

    details = "; ".join(errors) if errors else "no candidate"
    raise RuntimeError(
        f"Cannot resolve layer path `{path}` from model root `{type(model).__name__}`. Tried: {details}"
    )


_runner_uses_compilation_or_cudagraph = runner_uses_compilation_or_cudagraph
_infer_hidden_dtype = infer_hidden_dtype
_resolve_module_by_path_with_fallback = resolve_module_by_path_with_fallback


def _resolve_prompt_sample_for_req_id(req_id: str) -> tuple[int, int]:
    prompt_idx = RUNTIME.reqid_to_promptidx.get(req_id)
    if prompt_idx is not None:
        return int(prompt_idx), int(RUNTIME.reqid_to_sampleidx.get(req_id, 0))

    if "_" in req_id:
        maybe_sample_idx, parent_req_id = req_id.split("_", 1)
        if maybe_sample_idx.isdigit():
            parent_prompt_idx = RUNTIME.reqid_to_promptidx.get(parent_req_id)
            if parent_prompt_idx is not None:
                return int(parent_prompt_idx), int(maybe_sample_idx)
    return -1, -1


def _is_torch_compiling() -> bool:
    compiler = getattr(torch, "compiler", None)
    if compiler is not None:
        fn = getattr(compiler, "is_compiling", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        fn = getattr(dynamo, "is_compiling", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:
                pass

    return False


def configure_runtime(
    *,
    graph_scratch_rows: int,
    tap_layer_paths: Sequence[str],
    source_layer_path: str,
    target_layer_path: str,
    enable_esamp_training: bool,
    distiller_hidden_dim: int,
    distiller_lr: float,
    per_request_models: bool = False,
    per_request_model_bank: bool = False,
    model_bank_slots: int = 0,
    model_bank_flush_interval: int = 1,
    model_bank_rank: int = 64,
    model_bank_use_output_layernorm: bool = True,
    model_bank_initializer: object | None = None,
    model_bank_train_cudagraph: bool = False,
    model_bank_forward_backend: str = "torch",
    adaptation_pipeline_slots: int = 4,
    adaptation_stream_mode: str = "dual",
    adaptation_stream_priority: int = 0,
    trace_per_request_losses: bool = False,
    trace_interval: int = 1,
    trace_max_points: int = 0,
    enable_distiller_intervention: bool = False,
    distiller_beta: float = 0.0,
    distiller_sampler_backend: SamplerBackend = "post_filter_exact",
    compact_capture_lane: bool = False,
    inplace_ttt_fast_weight_enabled: bool = False,
    inplace_ttt_fast_weight_layers: Sequence[int] = (0,),
    inplace_ttt_fast_weight_chunk_size: int = 4096,
    inplace_ttt_fast_weight_lr: float = 1e-3,
    inplace_ttt_fast_weight_mode: str = "exact",
    inplace_ttt_fast_weight_decode_update_enabled: bool = False,
    inplace_ttt_fast_weight_key_mode: str = "static",
    inplace_ttt_fast_weight_static_request_id: str = "__tllm_inplace_ttt_static__",
    inplace_ttt_fast_weight_max_slots: int = 64,
    inplace_ttt_fast_weight_max_bytes: int = 0,
    inplace_ttt_fast_weight_fail_on_unsupported: bool = False,
) -> None:
    cfg = RUNTIME.config
    cfg.graph_scratch_rows = max(0, int(graph_scratch_rows))
    cfg.tap_layer_paths = [str(path).strip() for path in tap_layer_paths if str(path).strip()]
    cfg.source_layer_path = str(source_layer_path).strip()
    cfg.target_layer_path = str(target_layer_path).strip()
    cfg.enable_esamp_training = bool(enable_esamp_training)
    cfg.distiller_hidden_dim = max(1, int(distiller_hidden_dim))
    cfg.distiller_lr = float(distiller_lr)
    cfg.per_request_models = bool(per_request_models)
    cfg.per_request_model_bank = bool(per_request_model_bank)
    cfg.model_bank_slots = max(0, int(model_bank_slots))
    cfg.model_bank_flush_interval = max(1, int(model_bank_flush_interval))
    cfg.model_bank_rank = max(1, int(model_bank_rank))
    cfg.model_bank_use_output_layernorm = bool(model_bank_use_output_layernorm)
    cfg.model_bank_initializer = model_bank_initializer
    cfg.model_bank_train_cudagraph = bool(model_bank_train_cudagraph)
    cfg.model_bank_forward_backend = normalize_model_bank_forward_backend(model_bank_forward_backend)
    cfg.adaptation_pipeline_slots = max(1, int(adaptation_pipeline_slots))
    cfg.adaptation_stream_mode = normalize_adaptation_stream_mode(adaptation_stream_mode)
    cfg.adaptation_stream_priority = int(adaptation_stream_priority)
    cfg.trace_per_request_losses = bool(trace_per_request_losses)
    cfg.trace_interval = max(1, int(trace_interval))
    cfg.trace_max_points = max(0, int(trace_max_points))
    cfg.enable_distiller_intervention = bool(enable_distiller_intervention)
    cfg.distiller_beta = float(distiller_beta)
    cfg.distiller_sampler_backend = normalize_sampler_backend(str(distiller_sampler_backend).strip() or "post_filter_exact")
    cfg.compact_capture_lane = bool(compact_capture_lane)
    old_inplace_ttt_enabled = bool(cfg.inplace_ttt_fast_weight_enabled)
    cfg.inplace_ttt_fast_weight_enabled = bool(inplace_ttt_fast_weight_enabled)
    if old_inplace_ttt_enabled and not cfg.inplace_ttt_fast_weight_enabled:
        from tllm.runtime.inplace_ttt_fast_weight_runtime import restore_inplace_ttt_fast_weight_wrappers

        restore_inplace_ttt_fast_weight_wrappers(RUNTIME)
    cfg.inplace_ttt_fast_weight_layers = tuple(int(layer) for layer in inplace_ttt_fast_weight_layers)
    cfg.inplace_ttt_fast_weight_chunk_size = max(1, int(inplace_ttt_fast_weight_chunk_size))
    cfg.inplace_ttt_fast_weight_lr = float(inplace_ttt_fast_weight_lr)
    from tllm.runtime.fast_weight_state import normalize_fast_weight_mode

    cfg.inplace_ttt_fast_weight_mode = normalize_fast_weight_mode(inplace_ttt_fast_weight_mode)
    cfg.inplace_ttt_fast_weight_decode_update_enabled = bool(inplace_ttt_fast_weight_decode_update_enabled)
    cfg.inplace_ttt_fast_weight_key_mode = str(inplace_ttt_fast_weight_key_mode).strip() or "static"
    cfg.inplace_ttt_fast_weight_static_request_id = (
        str(inplace_ttt_fast_weight_static_request_id).strip() or "__tllm_inplace_ttt_static__"
    )
    cfg.inplace_ttt_fast_weight_max_slots = max(0, int(inplace_ttt_fast_weight_max_slots))
    cfg.inplace_ttt_fast_weight_max_bytes = max(0, int(inplace_ttt_fast_weight_max_bytes))
    cfg.inplace_ttt_fast_weight_fail_on_unsupported = bool(inplace_ttt_fast_weight_fail_on_unsupported)

    if not cfg.source_layer_path:
        raise RuntimeError("source_layer_path must be non-empty")
    if not cfg.target_layer_path:
        raise RuntimeError("target_layer_path must be non-empty")

    from tllm.runtime.ports.residual_bindings import default_raw_paths_from_config

    RUNTIME.residual_raw_paths = default_raw_paths_from_config(cfg)
    precompute = RUNTIME.sampler_precompute
    precompute.timing_enabled = os.getenv("TLLM_TRACE_DISTILLER_TIMING", "") == "1"
    precompute.precompute_event_pairs = []
    precompute.wait_event_pairs = []
    precompute.fallback_event_pairs = []
    precompute.port_publish_attempt_count = 0
    precompute.port_publish_hit_count = 0
    precompute.schedule_attempt_count = 0
    precompute.schedule_hit_count = 0
    RUNTIME.path_hotspot_enabled = os.getenv("TLLM_TRACE_PATH_HOTSPOTS", "") == "1"
    RUNTIME.path_hotspot_cpu_ms = {}
    RUNTIME.path_hotspot_counts = {}
    _refresh_dispatch_plan()
    RUNTIME.reset_step()


def set_esamp_training_enabled(enabled: bool) -> None:
    RUNTIME.config.enable_esamp_training = bool(enabled)
    if RUNTIME.consumer is not None:
        RUNTIME.consumer.set_enabled(enabled)


def set_test_time_lora_enabled(enabled: bool) -> None:
    RUNTIME.test_time_lora_enabled = bool(enabled)


def set_inplace_ttt_fast_weight_enabled(
    enabled: bool,
    *,
    layers: Sequence[int] | None = None,
    chunk_size: int | None = None,
    learning_rate: float | None = None,
    mode: str | None = None,
    decode_update_enabled: bool | None = None,
    key_mode: str | None = None,
    max_slots: int | None = None,
    max_bytes: int | None = None,
    fail_on_unsupported: bool | None = None,
) -> None:
    normalized_mode = None
    if mode is not None:
        from tllm.runtime.fast_weight_state import normalize_fast_weight_mode

        normalized_mode = normalize_fast_weight_mode(mode)
    RUNTIME.config.inplace_ttt_fast_weight_enabled = bool(enabled)
    if not bool(enabled):
        from tllm.runtime.inplace_ttt_fast_weight_runtime import restore_inplace_ttt_fast_weight_wrappers

        restore_inplace_ttt_fast_weight_wrappers(RUNTIME)
        RUNTIME.inplace_ttt_current_token_batch_meta = None
        RUNTIME.inplace_ttt_fast_weight_phase = "prefill"
    if layers is not None:
        RUNTIME.config.inplace_ttt_fast_weight_layers = tuple(int(layer) for layer in layers)
    if chunk_size is not None:
        RUNTIME.config.inplace_ttt_fast_weight_chunk_size = max(1, int(chunk_size))
    if learning_rate is not None:
        RUNTIME.config.inplace_ttt_fast_weight_lr = float(learning_rate)
    if normalized_mode is not None:
        RUNTIME.config.inplace_ttt_fast_weight_mode = normalized_mode
    if decode_update_enabled is not None:
        RUNTIME.config.inplace_ttt_fast_weight_decode_update_enabled = bool(decode_update_enabled)
    if key_mode is not None:
        RUNTIME.config.inplace_ttt_fast_weight_key_mode = str(key_mode).strip() or "static"
    if max_slots is not None:
        RUNTIME.config.inplace_ttt_fast_weight_max_slots = max(0, int(max_slots))
    if max_bytes is not None:
        RUNTIME.config.inplace_ttt_fast_weight_max_bytes = max(0, int(max_bytes))
    if fail_on_unsupported is not None:
        RUNTIME.config.inplace_ttt_fast_weight_fail_on_unsupported = bool(fail_on_unsupported)


def synchronize_esamp() -> None:
    if RUNTIME.consumer is not None:
        synchronize_consumer(RUNTIME.consumer)
    registry = RUNTIME.consumer_registry
    if registry is None:
        return
    for consumer in registry.consumers():
        synchronize_consumer(consumer)


def read_and_reset_esamp_stats(sync: bool = True) -> TrainingStats:
    if RUNTIME.consumer is None:
        return TrainingStats(loss_avg=0.0, loss_count=0)
    return RUNTIME.consumer.read_and_reset_stats(sync=sync)


def read_and_reset_esamp_per_request_stats(sync: bool = True) -> Dict[int, TrainingStats]:
    if RUNTIME.consumer is None:
        return {}
    return RUNTIME.consumer.read_and_reset_per_request_stats(sync=sync)


def read_and_reset_distiller_timing_stats(sync: bool = True) -> DistillerTimingStats:
    return _stats.read_and_reset_distiller_timing_stats(RUNTIME, sync=sync)


def record_path_hotspot_cpu(name: str, ms: float) -> None:
    _stats.record_path_hotspot_cpu(RUNTIME, name, ms)


def read_and_reset_path_hotspot_stats(sync: bool = True) -> PathHotspotStats:
    return _stats.read_and_reset_path_hotspot_stats(RUNTIME, sync=sync)


def read_graph_debug_stats(*, mode: str = "model_bank") -> GraphDebugStats:
    return _stats.read_graph_debug_stats(getattr(RUNTIME, "consumer", None), mode=mode)


def _prepare_decode_localization(runner: RunnerLike, out: tuple[object, ...]) -> None:
    _hooks.prepare_decode_localization(core=sys.modules[__name__], runner=runner, out=out)


def _build_tap_path_list() -> list[str]:
    return _hooks.build_tap_path_list(core=sys.modules[__name__])


def _setup_runtime_hooks_if_active(runner: RunnerLike) -> None:
    _hooks.setup_runtime_hooks_if_active(core=sys.modules[__name__], runner=runner)


def _wrapped_load_model(self: object, *args: object, **kwargs: object) -> object:
    return _hooks.wrapped_load_model(core=sys.modules[__name__], runner=self, args=args, kwargs=kwargs)


def _wrapped_prepare_inputs(self: object, scheduler_output: object) -> object:
    return _hooks.wrapped_prepare_inputs(core=sys.modules[__name__], runner=self, scheduler_output=scheduler_output)


def _wrapped_execute_model(self: object, *args: object, **kwargs: object) -> object:
    return _hooks.wrapped_execute_model(core=sys.modules[__name__], runner=self, args=args, kwargs=kwargs)


def install_runner_patch() -> None:
    _lifecycle.install_runner_patch(sys.modules[__name__])


def make_llm(*args: object, **kwargs: object) -> object:
    return _lifecycle.make_llm(sys.modules[__name__], *args, **kwargs)


__all__ = [
    "DistillerTimingStats",
    "ESampStats",
    "TrainingStats",
    "GraphDebugStats",
    "PathHotspotStats",
    "MODEL_HOOK_FLAG",
    "MODEL_HOOK_SPEC_ATTR",
    "RUNTIME",
    "ResidualRuntimeConfig",
    "ResidualRuntimeState",
    "RuntimeConsumer",
    "SamplerPrecomputeBuffers",
    "SamplerPrecomputeCache",
    "SamplerPrecomputeState",
    "clear_dispatch_consumers",
    "compute_decode_localization",
    "configure_runtime",
    "infer_hidden_dtype",
    "make_llm",
    "pick_common_attn_metadata",
    "read_and_reset_distiller_timing_stats",
    "read_and_reset_path_hotspot_stats",
    "read_graph_debug_stats",
    "read_and_reset_esamp_per_request_stats",
    "read_and_reset_esamp_stats",
    "register_dispatch_consumer",
    "replace_dispatch_consumers",
    "resolve_module_by_path_with_fallback",
    "record_path_hotspot_cpu",
    "runner_uses_compilation_or_cudagraph",
    "set_runtime_consumer",
    "set_esamp_training_enabled",
    "set_inplace_ttt_fast_weight_enabled",
    "set_test_time_lora_enabled",
    "synchronize_esamp",
]
