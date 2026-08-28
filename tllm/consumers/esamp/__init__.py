#!/usr/bin/env python3
"""Public ESamp consumer surface (lazy: submodule imports stay minimal)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "ESampConsumer",
    "activate",
    "ESampConsumerConfig",
    "ESampProgram",
    "ESampProgramConsumer",
    "ESampRegionConfig",
    "ESampSamplerModifierProvider",
    "build_esamp_workload",
    "ESampRegionWorkload",
]

_MODULE_BY_NAME = {
    "ESampConsumer": "tllm.consumers.esamp.consumer",
    "activate": "tllm.consumers.esamp.register",
    "ESampConsumerConfig": "tllm.consumers.esamp.config",
    "ESampProgram": "tllm.consumers.esamp.program",
    "ESampProgramConsumer": "tllm.consumers.esamp.program_consumer",
    "ESampRegionConfig": "tllm.consumers.esamp.region_workload",
    "ESampSamplerModifierProvider": "tllm.consumers.esamp.sampler_provider",
    "build_esamp_workload": "tllm.consumers.esamp.workload",
    "ESampRegionWorkload": "tllm.consumers.esamp.region_workload",
}


def __getattr__(name: str) -> Any:
    module_path = _MODULE_BY_NAME.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value
