#!/usr/bin/env python3
"""Shared, optional runtime diagnostics with no planning semantics."""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

from torch.autograd.profiler import record_function

_DISTILLER_PROFILING_ENABLED = os.getenv("TLLM_ENABLE_DISTILLER_RECORD_FUNCTION", "") == "1"


def maybe_record_function(name: str) -> AbstractContextManager[object]:
    if _DISTILLER_PROFILING_ENABLED:
        return record_function(name)
    return nullcontext()


class PathHotspotRuntimeLike(Protocol):
    path_hotspot_enabled: bool


class PathHotspotCoreLike(Protocol):
    @property
    def RUNTIME(self) -> PathHotspotRuntimeLike: ...

    def record_path_hotspot_cpu(self, name: str, ms: float) -> None: ...


def path_hotspots_enabled(core: PathHotspotCoreLike) -> bool:
    return bool(core.RUNTIME.path_hotspot_enabled)


def record_path_hotspot_cpu(
    core: PathHotspotCoreLike,
    name: str,
    start_s: float,
    end_s: float,
) -> None:
    if path_hotspots_enabled(core):
        core.record_path_hotspot_cpu(name, float((end_s - start_s) * 1000.0))
