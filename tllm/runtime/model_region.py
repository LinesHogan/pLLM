#!/usr/bin/env python3
"""Backend-neutral lifecycle for installed model-region executors."""

from __future__ import annotations

from typing import Protocol

from tllm.backends.protocol import RunnerBinding


class _ModelRegionRuntimeLike(Protocol):
    model_region_binding: object | None


def restore_model_region_executors(runtime: _ModelRegionRuntimeLike) -> int:
    binding = runtime.model_region_binding
    if not isinstance(binding, RunnerBinding):
        return 0
    restored = int(binding.driver.restore_model_region_executors(binding))
    runtime.model_region_binding = None
    return restored


__all__ = ["restore_model_region_executors"]
