#!/usr/bin/env python3
"""Public ESamp consumer surface (lazy: submodule imports stay minimal)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tllm.consumers.esamp.config import ESampConsumerConfig
    from tllm.consumers.esamp.sampler_provider import ESampSamplerModifierProvider
    from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload

__all__ = [
    "ESampConsumerConfig",
    "ESampConfig",
    "ESampSamplerModifierProvider",
    "ESampWorkload",
]

_MODULE_BY_NAME = {
    "ESampConsumerConfig": "tllm.consumers.esamp.config",
    "ESampConfig": "tllm.consumers.esamp.workload",
    "ESampSamplerModifierProvider": "tllm.consumers.esamp.sampler_provider",
    "ESampWorkload": "tllm.consumers.esamp.workload",
}


def __getattr__(name: str) -> Any:
    module_path = _MODULE_BY_NAME.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value
