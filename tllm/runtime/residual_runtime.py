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
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import torch

from tllm.common.state import ensure_v1_env, pick_common_attn_metadata, resolve_object_by_path
from tllm.contracts.consumer_stats import TrainingStats, merge_training_stats_maps
from tllm.contracts.request_identity import resolve_prompt_sample

# Backward-compatible alias: ESamp's stats name, canonical type is TrainingStats.
ESampStats = TrainingStats
from tllm.contracts.runtime_context import RunnerLike
from tllm.common.decode_localization import compute_decode_localization
from tllm.runtime import lifecycle as _lifecycle
from tllm.runtime import stats as _stats
from tllm.runtime.config import normalize_adaptation_stream_mode
from tllm.runtime.resources import (
    SamplerPrecomputeBuffers,
    SamplerPrecomputeCache,
    SamplerPrecomputeState,
)
from tllm.runtime.state import (
    RUNTIME,
    ResidualRuntimeConfig,
    ResidualRuntimeState,
)
from tllm.runtime.stats import DistillerTimingStats, GraphDebugStats, PathHotspotStats, PlacementStats

if TYPE_CHECKING:
    from vllm import LLM

ensure_v1_env()

GPUModelRunner = None
_ORIG_LOAD_MODEL = None
_ORIG_PREPARE_INPUTS = None
_ORIG_EXECUTE_MODEL = None
_PATCH_INSTALLED = False

MODEL_HOOK_FLAG = "esamp_hook_installed"
MODEL_HOOK_SPEC_ATTR = "esamp_hook_spec"


def runtime_patch_installed() -> bool:
    return _PATCH_INSTALLED


def advance_workload_anchor(anchor: str) -> bool:
    return False


def runner_uses_compilation_or_cudagraph(runner: RunnerLike) -> bool:
    """Legacy predicate kept for call sites; semantics now live in
    ``tllm.runtime.graph_policy.resolve_graph_mode`` (PIECEWISE/FULL ->
    True)."""
    from tllm.runtime import graph_policy as _graph_policy

    mode = _graph_policy.resolve_graph_mode(runner)
    if mode != _graph_policy.GraphMode.EAGER:
        return True

    from tllm.runtime.vllm_adapter import legacy_runner_graph_enabled

    return legacy_runner_graph_enabled(runner)


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
    return resolve_prompt_sample(
        req_id,
        prompt_indices=RUNTIME.reqid_to_promptidx,
        sample_indices=RUNTIME.reqid_to_sampleidx,
    )


def configure_runtime(
    *,
    graph_scratch_rows: int,
    tap_layer_paths: Sequence[str],
    source_layer_path: str,
    target_layer_path: str,
    adaptation_pipeline_slots: int = 4,
    adaptation_stream_mode: str = "dual",
    adaptation_stream_priority: int = 0,
    compact_capture_lane: bool = False,
) -> None:
    """Configure runtime-owned mechanisms (tap topology, streams, capture lanes).

    Consumer-algorithm settings (distiller dims/lr/beta, model bank policy,
    training toggles, tracing) are NOT accepted here — they belong to each
    consumer's own config, applied via the
    consumer's support workflow.
    """
    cfg = RUNTIME.config
    cfg.graph_scratch_rows = max(0, int(graph_scratch_rows))
    cfg.tap_layer_paths = [str(path).strip() for path in tap_layer_paths if str(path).strip()]
    cfg.source_layer_path = str(source_layer_path).strip()
    cfg.target_layer_path = str(target_layer_path).strip()
    cfg.adaptation_pipeline_slots = max(1, int(adaptation_pipeline_slots))
    cfg.adaptation_stream_mode = normalize_adaptation_stream_mode(adaptation_stream_mode)
    cfg.adaptation_stream_priority = int(adaptation_stream_priority)
    cfg.compact_capture_lane = bool(compact_capture_lane)

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
    RUNTIME.reset_step()


def set_esamp_training_enabled(enabled: bool) -> None:
    from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec
    from tllm.runtime.mp.control import active_control_plane

    refresh_exported_bootstrap_spec()
    plane = active_control_plane()
    if plane is not None:
        plane.apply_enable_flags({"esamp_training_enabled": bool(enabled)})


def synchronize_esamp() -> None:
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is not None:
        plane.synchronize()


def read_and_reset_esamp_stats(sync: bool = True) -> TrainingStats:
    local = TrainingStats(loss_avg=0.0, loss_count=0)
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return TrainingStats.merge([local, *(reply.esamp for reply in plane.pull_stats("esamp", sync=sync))])


def read_and_reset_esamp_per_request_stats(sync: bool = True) -> Dict[int, TrainingStats]:
    local: Dict[int, TrainingStats] = {}
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return merge_training_stats_maps(
        [local, *(reply.esamp_per_request for reply in plane.pull_stats("esamp_per_request", sync=sync))]
    )


def read_and_reset_distiller_timing_stats(sync: bool = True) -> DistillerTimingStats:
    local = _stats.read_and_reset_distiller_timing_stats(RUNTIME, sync=sync)
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return DistillerTimingStats.merge(
        [local, *(reply.distiller_timing for reply in plane.pull_stats("distiller_timing", sync=sync))]
    )


def record_path_hotspot_cpu(name: str, ms: float) -> None:
    _stats.record_path_hotspot_cpu(RUNTIME, name, ms)


def read_and_reset_path_hotspot_stats(sync: bool = True) -> PathHotspotStats:
    local = _stats.read_and_reset_path_hotspot_stats(RUNTIME, sync=sync)
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return PathHotspotStats.merge(
        [local, *(reply.path_hotspot for reply in plane.pull_stats("path_hotspot", sync=sync))]
    )


def read_and_reset_placement_stats() -> PlacementStats:
    local = _stats.read_and_reset_placement_stats(RUNTIME)
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    if plane is None:
        return local
    return PlacementStats.merge([local, *(reply.placement for reply in plane.pull_stats("placement"))])


def read_graph_debug_stats(*, mode: str = "model_bank") -> GraphDebugStats:
    return _stats.read_graph_debug_stats(None, mode=mode)





def _wrapped_load_model(self: object, *args: object, **kwargs: object) -> object:
    from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks

    return _hooks.wrapped_load_model(core=sys.modules[__name__], runner=self, args=args, kwargs=kwargs)


def _wrapped_prepare_inputs(self: object, scheduler_output: object) -> object:
    from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks

    return _hooks.wrapped_prepare_inputs(core=sys.modules[__name__], runner=self, scheduler_output=scheduler_output)


def _wrapped_execute_model(self: object, *args: object, **kwargs: object) -> object:
    from tllm.runtime.vllm_patch import port_runtime_hooks as _hooks

    return _hooks.wrapped_execute_model(core=sys.modules[__name__], runner=self, args=args, kwargs=kwargs)


def install_runner_patch() -> None:
    _lifecycle.install_runner_patch(sys.modules[__name__])


def make_llm(*args: object, **kwargs: object) -> "LLM":
    return _lifecycle.make_llm(sys.modules[__name__], *args, **kwargs)


__all__ = [
    "DistillerTimingStats",
    "ESampStats",
    "TrainingStats",
    "GraphDebugStats",
    "PathHotspotStats",
    "PlacementStats",
    "MODEL_HOOK_FLAG",
    "MODEL_HOOK_SPEC_ATTR",
    "RUNTIME",
    "ResidualRuntimeConfig",
    "ResidualRuntimeState",
    "SamplerPrecomputeBuffers",
    "SamplerPrecomputeCache",
    "SamplerPrecomputeState",
    "compute_decode_localization",
    "configure_runtime",
    "infer_hidden_dtype",
    "make_llm",
    "pick_common_attn_metadata",
    "read_and_reset_distiller_timing_stats",
    "read_and_reset_path_hotspot_stats",
    "read_and_reset_placement_stats",
    "read_graph_debug_stats",
    "read_and_reset_esamp_per_request_stats",
    "read_and_reset_esamp_stats",
    "resolve_module_by_path_with_fallback",
    "record_path_hotspot_cpu",
    "runner_uses_compilation_or_cudagraph",
    "set_esamp_training_enabled",
    "synchronize_esamp",
]
