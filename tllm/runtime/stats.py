#!/usr/bin/env python3
"""Runtime statistics containers and collectors.

Pure reporting layer: the read/reset helpers take the runtime state (and,
where needed, the active consumer) explicitly, so nothing here imports the
hook or lifecycle machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch


@dataclass
class DistillerTimingStats:
    port_publish_attempt_count: int = 0
    port_publish_hit_count: int = 0
    precompute_ms_total: float = 0.0
    precompute_count: int = 0
    wait_ms_total: float = 0.0
    wait_count: int = 0
    schedule_attempt_count: int = 0
    schedule_hit_count: int = 0
    fallback_ms_total: float = 0.0
    fallback_count: int = 0
    candidate_sample_count: int = 0
    candidate_token_count: int = 0
    candidate_row_count: int = 0
    candidate_max_count: int = 0
    candidate_kernel_triton_count: int = 0
    candidate_kernel_torch_count: int = 0
    candidate_kernel_fallback_count: int = 0

    @property
    def precompute_ms_avg(self) -> float:
        return float(self.precompute_ms_total / self.precompute_count) if self.precompute_count > 0 else 0.0

    @property
    def wait_ms_avg(self) -> float:
        return float(self.wait_ms_total / self.wait_count) if self.wait_count > 0 else 0.0

    @property
    def fallback_ms_avg(self) -> float:
        return float(self.fallback_ms_total / self.fallback_count) if self.fallback_count > 0 else 0.0


@dataclass
class GraphDebugStats:
    capture_state: str = "uncaptured"
    capture_attempt_count: int = 0
    skip_not_enabled_count: int = 0
    skip_missing_optimizer_state_count: int = 0
    skip_wrong_device_count: int = 0
    replay_attempt_count: int = 0
    replay_hit_count: int = 0
    replay_stage_miss_count: int = 0
    kernel_fallback_count: int = 0
    disable_reason: str = ""


@dataclass
class PathHotspotStats:
    cpu_ms_total: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


def read_and_reset_distiller_timing_stats(state: object, sync: bool = True) -> DistillerTimingStats:
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    out = DistillerTimingStats()

    def _accumulate(pairs: list[tuple[object, object]]) -> tuple[float, int]:
        total = 0.0
        count = 0
        for start, end in pairs:
            try:
                total += float(start.elapsed_time(end))
            except RuntimeError:
                continue
            count += 1
        return total, count

    precompute = state.sampler_precompute
    out.precompute_ms_total, out.precompute_count = _accumulate(precompute.precompute_event_pairs)
    out.wait_ms_total, out.wait_count = _accumulate(precompute.wait_event_pairs)
    out.fallback_ms_total, out.fallback_count = _accumulate(precompute.fallback_event_pairs)
    out.port_publish_attempt_count = int(precompute.port_publish_attempt_count)
    out.port_publish_hit_count = int(precompute.port_publish_hit_count)
    out.schedule_attempt_count = int(precompute.schedule_attempt_count)
    out.schedule_hit_count = int(precompute.schedule_hit_count)
    out.candidate_sample_count = int(precompute.candidate_sample_count)
    out.candidate_token_count = int(precompute.candidate_token_count)
    out.candidate_row_count = int(precompute.candidate_row_count)
    out.candidate_max_count = int(precompute.candidate_max_count)
    out.candidate_kernel_triton_count = int(precompute.candidate_kernel_triton_count)
    out.candidate_kernel_torch_count = int(precompute.candidate_kernel_torch_count)
    out.candidate_kernel_fallback_count = int(precompute.candidate_kernel_fallback_count)
    precompute.precompute_event_pairs = []
    precompute.wait_event_pairs = []
    precompute.fallback_event_pairs = []
    precompute.port_publish_attempt_count = 0
    precompute.port_publish_hit_count = 0
    precompute.schedule_attempt_count = 0
    precompute.schedule_hit_count = 0
    precompute.candidate_sample_count = 0
    precompute.candidate_token_count = 0
    precompute.candidate_row_count = 0
    precompute.candidate_max_count = 0
    precompute.candidate_kernel_triton_count = 0
    precompute.candidate_kernel_torch_count = 0
    precompute.candidate_kernel_fallback_count = 0
    return out


def record_path_hotspot_cpu(state: object, name: str, ms: float) -> None:
    if not bool(getattr(state, "path_hotspot_enabled", False)):
        return
    key = str(name).strip()
    if not key:
        return
    state.path_hotspot_cpu_ms[key] = float(state.path_hotspot_cpu_ms.get(key, 0.0)) + float(ms)
    state.path_hotspot_counts[key] = int(state.path_hotspot_counts.get(key, 0)) + 1


def read_and_reset_path_hotspot_stats(state: object, sync: bool = True) -> PathHotspotStats:
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    out = PathHotspotStats(
        cpu_ms_total={str(k): float(v) for k, v in state.path_hotspot_cpu_ms.items()},
        counts={str(k): int(v) for k, v in state.path_hotspot_counts.items()},
    )
    state.path_hotspot_cpu_ms = {}
    state.path_hotspot_counts = {}
    return out


def read_graph_debug_stats(consumer: object, *, mode: str = "model_bank") -> GraphDebugStats:
    engine = getattr(consumer, "_engine", None)
    if engine is None:
        return GraphDebugStats()
    if str(mode) == "model_bank":
        slot_graphs = list(getattr(engine.state, "model_bank_slot_graphs", {}).values())
        active_slot_graphs = [graph for graph in slot_graphs if graph.capture_attempt_count or graph.replay_attempt_count or graph.capture_state != "uncaptured"]
        if active_slot_graphs:
            disabled = [graph for graph in active_slot_graphs if graph.capture_state == "disabled"]
            captured = [graph for graph in active_slot_graphs if graph.capture_state == "captured"]
            state = "disabled" if disabled else ("captured" if captured else "uncaptured")
            return GraphDebugStats(
                capture_state=state,
                capture_attempt_count=sum(int(graph.capture_attempt_count) for graph in active_slot_graphs),
                skip_not_enabled_count=sum(int(graph.skip_not_enabled_count) for graph in active_slot_graphs),
                skip_missing_optimizer_state_count=sum(int(graph.skip_missing_optimizer_state_count) for graph in active_slot_graphs),
                skip_wrong_device_count=sum(int(graph.skip_wrong_device_count) for graph in active_slot_graphs),
                replay_attempt_count=sum(int(graph.replay_attempt_count) for graph in active_slot_graphs),
                replay_hit_count=sum(int(graph.replay_hit_count) for graph in active_slot_graphs),
                replay_stage_miss_count=sum(int(graph.replay_stage_miss_count) for graph in active_slot_graphs),
                kernel_fallback_count=sum(int(graph.kernel_fallback_count) for graph in active_slot_graphs),
                disable_reason="; ".join(str(graph.disable_reason) for graph in disabled if str(graph.disable_reason)),
            )
    graph = engine.state.graphs.get(str(mode))
    if graph is None:
        return GraphDebugStats()
    return GraphDebugStats(
        capture_state=str(graph.capture_state),
        capture_attempt_count=int(graph.capture_attempt_count),
        skip_not_enabled_count=int(graph.skip_not_enabled_count),
        skip_missing_optimizer_state_count=int(graph.skip_missing_optimizer_state_count),
        skip_wrong_device_count=int(graph.skip_wrong_device_count),
        replay_attempt_count=int(graph.replay_attempt_count),
        replay_hit_count=int(graph.replay_hit_count),
        replay_stage_miss_count=int(graph.replay_stage_miss_count),
        kernel_fallback_count=int(graph.kernel_fallback_count),
        disable_reason=str(graph.disable_reason),
    )
