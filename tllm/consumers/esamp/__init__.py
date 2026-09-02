"""ESamp typed workload and independent tensor model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tllm.consumers.esamp.workload import ESampConfig, ESampWorkload

__all__ = [
    "ESampConfig",
    "ESampWorkload",
]

_MODULE_BY_NAME = {
    "ESampConfig": "tllm.consumers.esamp.workload",
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
