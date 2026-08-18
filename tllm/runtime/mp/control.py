#!/usr/bin/env python3
"""Parent-side control plane over vLLM's collective_rpc.

The plane invokes the string-named methods of ``TllmWorkerExtension`` on the
workers that own the model runner — in-process under mp=0 (InprocClient) and
in the EngineCore child under mp=1 (SyncMPClient). Using one mechanism for
both modes keeps the four parity cells on identical code paths; only string
method names cross the wire, so ``VLLM_ALLOW_INSECURE_SERIALIZATION`` is never
required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from tllm.contracts.consumer_stats import (
    TrainingStats,
    training_stats_map_from_wire,
)
from tllm.runtime.stats import DistillerTimingStats, PathHotspotStats, PlacementStats

TLLM_WORKER_EXTENSION_QUALNAME = "tllm.runtime.mp.worker_extension.TllmWorkerExtension"


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


class ControlPlane:
    """Thin facade: one method per worker-extension entry point."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    @staticmethod
    def try_bind(llm: Any) -> "ControlPlane | None":
        """Bind when the LLM exposes a collective_rpc-capable engine core.

        Test doubles and non-vLLM runners may lack one; callers then fall
        back to process-local behavior (mp=0 semantics).
        """
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
        return list(core_client.collective_rpc(str(method), timeout=timeout, args=tuple(args)))

    def ping(self, *, timeout: float | None = None) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self.call("tllm_ping", timeout=timeout)]

    def push_request_mapping(
        self, mappings: Iterable[tuple[str, int, int]], *, timeout: float | None = None
    ) -> None:
        rows = [[str(req_id), int(prompt_idx), int(sample_idx)] for req_id, prompt_idx, sample_idx in mappings]
        if not rows:
            return
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
            for entry in self.call("tllm_pull_stats", str(category), bool(sync), timeout=timeout)
        ]

    def synchronize(self, *, timeout: float | None = None) -> None:
        self.call("tllm_synchronize", timeout=timeout)

    def apply_enable_flags(self, flags: Mapping[str, bool], *, timeout: float | None = None) -> None:
        self.call("tllm_apply_enable_flags", {str(k): bool(v) for k, v in flags.items()}, timeout=timeout)

    def register_consumer(
        self, name: str, config: Mapping[str, Any], *, primary: bool = False, timeout: float | None = None
    ) -> None:
        self.call("tllm_register_consumer", str(name), dict(config), bool(primary), timeout=timeout)

    def clear_consumers(self, *, timeout: float | None = None) -> None:
        self.call("tllm_clear_consumers", timeout=timeout)

    def set_fast_weight_enabled(
        self,
        enabled: bool,
        *,
        layers: Iterable[int] | None = None,
        chunk_size: int | None = None,
        learning_rate: float | None = None,
        mode: str | None = None,
        decode_update_enabled: bool | None = None,
        key_mode: str | None = None,
        max_slots: int | None = None,
        max_bytes: int | None = None,
        fail_on_unsupported: bool | None = None,
        timeout: float | None = None,
    ) -> None:
        self.call(
            "tllm_set_fast_weight_enabled",
            bool(enabled),
            None if layers is None else [int(layer) for layer in layers],
            chunk_size,
            learning_rate,
            mode,
            decode_update_enabled,
            key_mode,
            max_slots,
            max_bytes,
            fail_on_unsupported,
            timeout=timeout,
        )
