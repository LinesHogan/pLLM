#!/usr/bin/env python3
"""Parent-side control plane over vLLM's collective_rpc.

This module owns the worker calls used by the residual vLLM integration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, TypedDict

from tllm.contracts.consumer_stats import (
    TrainingStats,
    training_stats_map_from_wire,
)
from tllm.runtime.stats import DistillerTimingStats, PathHotspotStats, PlacementStats

TLLM_WORKER_EXTENSION_QUALNAME = "tllm.runtime.mp.worker_extension.TllmWorkerExtension"


class WorkerPingStatus(TypedDict):
    pid: int
    worker_id: str
    patch_installed: bool
    mapping_size: int


@dataclass(frozen=True, slots=True)
class WorkerStatsReply:
    esamp: TrainingStats
    esamp_per_request: Mapping[int, TrainingStats]
    distiller_timing: DistillerTimingStats
    path_hotspot: PathHotspotStats
    placement: PlacementStats

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> "WorkerStatsReply":
        return cls(
            esamp=TrainingStats.from_wire(payload["esamp"]),
            esamp_per_request=training_stats_map_from_wire(payload["esamp_per_request"]),
            distiller_timing=DistillerTimingStats.from_wire(payload["distiller_timing"]),
            path_hotspot=PathHotspotStats.from_wire(payload["path_hotspot"]),
            placement=PlacementStats.from_wire(payload["placement"]),
        )


_ACTIVE_CONTROL_PLANE: "ControlPlane | None" = None


def set_active_control_plane(plane: "ControlPlane | None") -> None:
    global _ACTIVE_CONTROL_PLANE
    _ACTIVE_CONTROL_PLANE = plane


def active_control_plane() -> "ControlPlane | None":
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "0") != "1":
        return None
    return _ACTIVE_CONTROL_PLANE


def _worker_ping_from_wire(payload: object) -> WorkerPingStatus:
    if not isinstance(payload, Mapping):
        raise TypeError("worker ping reply must be a mapping")
    pid = payload.get("pid")
    worker_id = payload.get("worker_id")
    patch_installed = payload.get("patch_installed")
    mapping_size = payload.get("mapping_size")
    if type(pid) is not int or type(worker_id) is not str:
        raise TypeError("worker ping identity is malformed")
    if type(patch_installed) is not bool or type(mapping_size) is not int:
        raise TypeError("worker ping state is malformed")
    return {
        "pid": pid,
        "worker_id": worker_id,
        "patch_installed": patch_installed,
        "mapping_size": mapping_size,
    }


class ControlPlane:
    """Thin facade: one method per live worker-extension entry point."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    @staticmethod
    def try_bind(llm: Any) -> "ControlPlane | None":
        try:
            core_client = llm.llm_engine.engine_core
            if not callable(core_client.collective_rpc):
                return None
        except AttributeError:
            return None
        return ControlPlane(llm)

    def call(self, method: str, *args: Any, timeout: float | None = None) -> list[Any]:
        try:
            core_client = self._llm.llm_engine.engine_core
        except AttributeError as exc:
            raise ValueError("control plane requires an LLM with llm_engine") from exc
        return list(
            core_client.collective_rpc(str(method), timeout=timeout, args=tuple(args))
        )

    def ping(self, *, timeout: float | None = None) -> list[WorkerPingStatus]:
        return [
            _worker_ping_from_wire(entry)
            for entry in self.call("tllm_ping", timeout=timeout)
        ]

    def push_request_mapping(
        self,
        mappings: Iterable[tuple[str, int, int | None]],
        *,
        timeout: float | None = None,
    ) -> None:
        rows = [
            [
                str(request_id),
                int(prompt_idx),
                None if sample_idx is None else int(sample_idx),
            ]
            for request_id, prompt_idx, sample_idx in mappings
        ]
        if rows:
            self.call("tllm_push_request_mapping", rows, timeout=timeout)

    def pull_stats(
        self,
        category: str = "all",
        *,
        sync: bool = True,
        timeout: float | None = None,
    ) -> list[WorkerStatsReply]:
        return [
            WorkerStatsReply.from_wire(dict(entry))
            for entry in self.call(
                "tllm_pull_stats", str(category), bool(sync), timeout=timeout
            )
        ]

    def synchronize(self, *, timeout: float | None = None) -> None:
        self.call("tllm_synchronize", timeout=timeout)

    def apply_enable_flags(
        self, flags: Mapping[str, bool], *, timeout: float | None = None
    ) -> None:
        self.call("tllm_apply_enable_flags", dict(flags), timeout=timeout)
