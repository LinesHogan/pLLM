#!/usr/bin/env python3
"""Runtime statistics containers and collectors.

Pure reporting layer: the read/reset helpers take the runtime state (and,
where needed, the active consumer) explicitly, so nothing here imports the
hook or lifecycle machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Protocol

import torch

from tllm.runtime.resources import SamplerPrecomputeState


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

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "DistillerTimingStats":
        return cls(
            port_publish_attempt_count=int(payload.get("port_publish_attempt_count", 0)),
            port_publish_hit_count=int(payload.get("port_publish_hit_count", 0)),
            precompute_ms_total=float(payload.get("precompute_ms_total", 0.0)),
            precompute_count=int(payload.get("precompute_count", 0)),
            wait_ms_total=float(payload.get("wait_ms_total", 0.0)),
            wait_count=int(payload.get("wait_count", 0)),
            schedule_attempt_count=int(payload.get("schedule_attempt_count", 0)),
            schedule_hit_count=int(payload.get("schedule_hit_count", 0)),
            fallback_ms_total=float(payload.get("fallback_ms_total", 0.0)),
            fallback_count=int(payload.get("fallback_count", 0)),
            candidate_sample_count=int(payload.get("candidate_sample_count", 0)),
            candidate_token_count=int(payload.get("candidate_token_count", 0)),
            candidate_row_count=int(payload.get("candidate_row_count", 0)),
            candidate_max_count=int(payload.get("candidate_max_count", 0)),
            candidate_kernel_triton_count=int(payload.get("candidate_kernel_triton_count", 0)),
            candidate_kernel_torch_count=int(payload.get("candidate_kernel_torch_count", 0)),
            candidate_kernel_fallback_count=int(payload.get("candidate_kernel_fallback_count", 0)),
        )

    @classmethod
    def merge(cls, entries: Iterable["DistillerTimingStats"]) -> "DistillerTimingStats":
        out = cls()
        for entry in entries:
            out.port_publish_attempt_count += entry.port_publish_attempt_count
            out.port_publish_hit_count += entry.port_publish_hit_count
            out.precompute_ms_total += entry.precompute_ms_total
            out.precompute_count += entry.precompute_count
            out.wait_ms_total += entry.wait_ms_total
            out.wait_count += entry.wait_count
            out.schedule_attempt_count += entry.schedule_attempt_count
            out.schedule_hit_count += entry.schedule_hit_count
            out.fallback_ms_total += entry.fallback_ms_total
            out.fallback_count += entry.fallback_count
            out.candidate_sample_count += entry.candidate_sample_count
            out.candidate_token_count += entry.candidate_token_count
            out.candidate_row_count += entry.candidate_row_count
            out.candidate_max_count = max(out.candidate_max_count, entry.candidate_max_count)
            out.candidate_kernel_triton_count += entry.candidate_kernel_triton_count
            out.candidate_kernel_torch_count += entry.candidate_kernel_torch_count
            out.candidate_kernel_fallback_count += entry.candidate_kernel_fallback_count
        return out


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

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "PathHotspotStats":
        return cls(
            cpu_ms_total={str(key): float(value) for key, value in payload.get("cpu_ms_total", {}).items()},
            counts={str(key): int(value) for key, value in payload.get("counts", {}).items()},
        )

    @classmethod
    def merge(cls, entries: Iterable["PathHotspotStats"]) -> "PathHotspotStats":
        cpu_ms_total: dict[str, float] = {}
        counts: dict[str, int] = {}
        for entry in entries:
            for key, value in entry.cpu_ms_total.items():
                cpu_ms_total[key] = cpu_ms_total.get(key, 0.0) + float(value)
            for key, value in entry.counts.items():
                counts[key] = counts.get(key, 0) + int(value)
        return cls(cpu_ms_total=cpu_ms_total, counts=counts)


@dataclass
class PlacementStats:
    pinned_dispatch_count: int = 0
    moved_dispatch_count: int = 0
    enqueued_count: int = 0
    bubble_drain_count: int = 0
    deadline_flush_count: int = 0
    adaptive_run_count: int = 0
    adaptive_defer_count: int = 0
    adaptive_fallback_count: int = 0
    adaptive_shed_count: int = 0
    max_queue_depth: int = 0
    bubble_drain_ms: float = 0.0
    fusion_eligible_count: int = 0
    fusion_fallback_count: int = 0
    fusion_execution_count: int = 0
    fusion_replay_count: int = 0
    fusion_eager_fallback_count: int = 0
    producer_stage_dispatch_count: int = 0
    capture_resident_stage_count: int = 0
    capture_resident_stage_install_count: int = 0
    capture_resident_stage_dispatch_count: int = 0
    step_lease_retire_count: int = 0

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "PlacementStats":
        return cls(
            pinned_dispatch_count=int(payload.get("pinned_dispatch_count", 0)),
            moved_dispatch_count=int(payload.get("moved_dispatch_count", 0)),
            enqueued_count=int(payload.get("enqueued_count", 0)),
            bubble_drain_count=int(payload.get("bubble_drain_count", 0)),
            deadline_flush_count=int(payload.get("deadline_flush_count", 0)),
            adaptive_run_count=int(payload.get("adaptive_run_count", 0)),
            adaptive_defer_count=int(payload.get("adaptive_defer_count", 0)),
            adaptive_fallback_count=int(payload.get("adaptive_fallback_count", 0)),
            adaptive_shed_count=int(payload.get("adaptive_shed_count", 0)),
            max_queue_depth=int(payload.get("max_queue_depth", 0)),
            bubble_drain_ms=float(payload.get("bubble_drain_ms", 0.0)),
            fusion_eligible_count=int(payload.get("fusion_eligible_count", 0)),
            fusion_fallback_count=int(payload.get("fusion_fallback_count", 0)),
            fusion_execution_count=int(payload.get("fusion_execution_count", 0)),
            fusion_replay_count=int(payload.get("fusion_replay_count", 0)),
            fusion_eager_fallback_count=int(payload.get("fusion_eager_fallback_count", 0)),
            producer_stage_dispatch_count=int(payload.get("producer_stage_dispatch_count", 0)),
            capture_resident_stage_count=int(payload.get("capture_resident_stage_count", 0)),
            capture_resident_stage_install_count=int(
                payload.get("capture_resident_stage_install_count", 0)
            ),
            capture_resident_stage_dispatch_count=int(
                payload.get("capture_resident_stage_dispatch_count", 0)
            ),
            step_lease_retire_count=int(payload.get("step_lease_retire_count", 0)),
        )

    @classmethod
    def merge(cls, entries: Iterable["PlacementStats"]) -> "PlacementStats":
        out = cls()
        for entry in entries:
            out.pinned_dispatch_count += entry.pinned_dispatch_count
            out.moved_dispatch_count += entry.moved_dispatch_count
            out.enqueued_count += entry.enqueued_count
            out.bubble_drain_count += entry.bubble_drain_count
            out.deadline_flush_count += entry.deadline_flush_count
            out.adaptive_run_count += entry.adaptive_run_count
            out.adaptive_defer_count += entry.adaptive_defer_count
            out.adaptive_fallback_count += entry.adaptive_fallback_count
            out.adaptive_shed_count += entry.adaptive_shed_count
            out.max_queue_depth = max(out.max_queue_depth, entry.max_queue_depth)
            out.bubble_drain_ms += entry.bubble_drain_ms
            out.fusion_eligible_count += entry.fusion_eligible_count
            out.fusion_fallback_count += entry.fusion_fallback_count
            out.fusion_execution_count += entry.fusion_execution_count
            out.fusion_replay_count += entry.fusion_replay_count
            out.fusion_eager_fallback_count += entry.fusion_eager_fallback_count
            out.producer_stage_dispatch_count += entry.producer_stage_dispatch_count
            out.capture_resident_stage_count = max(
                out.capture_resident_stage_count,
                entry.capture_resident_stage_count,
            )
            out.capture_resident_stage_install_count += (
                entry.capture_resident_stage_install_count
            )
            out.capture_resident_stage_dispatch_count += (
                entry.capture_resident_stage_dispatch_count
            )
            out.step_lease_retire_count += entry.step_lease_retire_count
        return out


class StatsState(Protocol):
    sampler_precompute: SamplerPrecomputeState
    path_hotspot_enabled: bool
    path_hotspot_cpu_ms: dict[str, float]
    path_hotspot_counts: dict[str, int]
    placement_stats: PlacementStats


def read_and_reset_distiller_timing_stats(state: StatsState, sync: bool = True) -> DistillerTimingStats:
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    out = DistillerTimingStats()

    def _accumulate(pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> tuple[float, int]:
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


def record_path_hotspot_cpu(state: StatsState, name: str, ms: float) -> None:
    if not state.path_hotspot_enabled:
        return
    key = str(name).strip()
    if not key:
        return
    state.path_hotspot_cpu_ms[key] = float(state.path_hotspot_cpu_ms.get(key, 0.0)) + float(ms)
    state.path_hotspot_counts[key] = int(state.path_hotspot_counts.get(key, 0)) + 1


def read_and_reset_path_hotspot_stats(state: StatsState, sync: bool = True) -> PathHotspotStats:
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    out = PathHotspotStats(
        cpu_ms_total={str(k): float(v) for k, v in state.path_hotspot_cpu_ms.items()},
        counts={str(k): int(v) for k, v in state.path_hotspot_counts.items()},
    )
    state.path_hotspot_cpu_ms = {}
    state.path_hotspot_counts = {}
    return out


def read_and_reset_placement_stats(state: StatsState) -> PlacementStats:
    current = state.placement_stats
    out = PlacementStats(
        pinned_dispatch_count=int(current.pinned_dispatch_count),
        moved_dispatch_count=int(current.moved_dispatch_count),
        enqueued_count=int(current.enqueued_count),
        bubble_drain_count=int(current.bubble_drain_count),
        deadline_flush_count=int(current.deadline_flush_count),
        adaptive_run_count=int(current.adaptive_run_count),
        adaptive_defer_count=int(current.adaptive_defer_count),
        adaptive_fallback_count=int(current.adaptive_fallback_count),
        adaptive_shed_count=int(current.adaptive_shed_count),
        max_queue_depth=int(current.max_queue_depth),
        bubble_drain_ms=float(current.bubble_drain_ms),
        fusion_eligible_count=int(current.fusion_eligible_count),
        fusion_fallback_count=int(current.fusion_fallback_count),
        fusion_execution_count=int(current.fusion_execution_count),
        fusion_replay_count=int(current.fusion_replay_count),
        fusion_eager_fallback_count=int(current.fusion_eager_fallback_count),
        producer_stage_dispatch_count=int(current.producer_stage_dispatch_count),
        capture_resident_stage_count=int(current.capture_resident_stage_count),
        capture_resident_stage_install_count=int(
            current.capture_resident_stage_install_count
        ),
        capture_resident_stage_dispatch_count=int(
            current.capture_resident_stage_dispatch_count
        ),
        step_lease_retire_count=int(current.step_lease_retire_count),
    )
    state.placement_stats = PlacementStats()
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
