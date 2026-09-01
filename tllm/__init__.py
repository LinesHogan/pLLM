"""Public tLLM API and the sole clean-room/legacy routing boundary.

Nominal clean-room workloads are routed before the legacy runtime is imported.
All other module-level calls remain a backward-compatible façade over the
process-wide default :class:`tllm.runtime.runtime.TllmRuntime`:

    runtime = TllmRuntime(consumers=[...])
    llm = runtime.make_llm(...)

The legacy/runtime names (:class:`BaseConsumer`, :class:`TllmRuntime`,
``default_runtime``, ``BeforeNextUse``, ``extern``, ``flow``) load lazily via
PEP 562 so that importing a native-stack subpackage (``tllm.workload``,
``tllm.schedule``) never pulls the legacy Program/workgraph frontend or the
runtime into a compiler-only process.  They are gone at legacy retirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tllm.consumers.base import BaseConsumer
    from tllm.runtime.runtime import TllmRuntime, default_runtime
    from tllm.workgraph.frontend import BeforeNextUse, extern, flow


def make_llm(*args: object, **kwargs: object) -> object:
    workloads = kwargs.pop("workloads", None)
    workload_items: tuple[object, ...] = ()
    if workloads is not None:
        if not isinstance(workloads, (tuple, list)):
            raise TypeError("make_llm.workloads must be a tuple or list")
        workload_items = tuple(workloads)
        if workload_items:
            from tllm.canonical.workloads.base import CanonicalWorkload

            canonical = tuple(
                isinstance(workload, CanonicalWorkload)
                for workload in workload_items
            )
            if any(canonical):
                if not all(canonical):
                    raise TypeError(
                        "canonical and legacy workloads cannot share one runtime"
                    )
                return _make_canonical_llm(args, kwargs, workload_items)

    from tllm.runtime.runtime import TllmRuntime, default_runtime

    runtime_option_names = (
        "workload_memory_budget_bytes",
        "workload_allow_serial_fallback",
        "workload_persistent_state_bytes",
        "workload_decision_cache_directory",
    )
    runtime_options = {
        name: kwargs.pop(name) for name in runtime_option_names if name in kwargs
    }
    if workloads is not None:
        memory_budget = runtime_options.get("workload_memory_budget_bytes", 0)
        allow_serial = runtime_options.get("workload_allow_serial_fallback", True)
        persistent_state = runtime_options.get("workload_persistent_state_bytes", 0)
        cache_directory = runtime_options.get("workload_decision_cache_directory", None)
        if not isinstance(memory_budget, int) or isinstance(memory_budget, bool):
            raise TypeError("workload_memory_budget_bytes must be int")
        if not isinstance(allow_serial, bool):
            raise TypeError("workload_allow_serial_fallback must be bool")
        if not isinstance(persistent_state, int) or isinstance(persistent_state, bool):
            raise TypeError("workload_persistent_state_bytes must be int")
        if cache_directory is not None and not (
            isinstance(cache_directory, str) and cache_directory
        ):
            raise TypeError("workload_decision_cache_directory must be str or None")
        return TllmRuntime(
            workloads=workload_items,
            workload_memory_budget_bytes=memory_budget,
            workload_allow_serial_fallback=allow_serial,
            workload_persistent_state_bytes=persistent_state,
            workload_decision_cache_directory=cache_directory,
        ).make_llm(*args, **kwargs)
    if runtime_options:
        raise TypeError("workload compiler options require make_llm(workloads=...)")
    return default_runtime().make_llm(*args, **kwargs)


def _make_canonical_llm(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    workloads: tuple[object, ...],
) -> object:
    """The sole dynamic coexistence boundary; canonical internals stay closed."""

    from tllm.canonical.workloads.esamp_request import ESampRequestWorkload
    from tllm.canonical_runtime.runtime import (
        CanonicalVllmConfig,
        create_canonical_esamp_llm,
    )

    if len(workloads) != 1 or type(workloads[0]) is not ESampRequestWorkload:
        raise TypeError("canonical vLLM currently supports one ESamp request workload")
    workload = workloads[0]
    if args or set(kwargs) != {"canonical_config"}:
        raise TypeError(
            "canonical make_llm requires only canonical_config=CanonicalVllmConfig(...)"
        )
    config = kwargs["canonical_config"]
    if type(config) is not CanonicalVllmConfig:
        raise TypeError("canonical_config must be an exact CanonicalVllmConfig")
    return create_canonical_esamp_llm(workload, config)


def register_consumer(consumer: BaseConsumer) -> None:
    from tllm.runtime.runtime import default_runtime

    default_runtime().register_consumer(consumer)


def clear_consumers() -> None:
    from tllm.runtime.runtime import default_runtime

    default_runtime().clear_consumers()


def __getattr__(name: str) -> object:
    if name == "BaseConsumer":
        from tllm.consumers.base import BaseConsumer

        return BaseConsumer
    if name in ("TllmRuntime", "default_runtime"):
        from tllm.runtime import runtime as _runtime

        return getattr(_runtime, name)
    if name in ("BeforeNextUse", "extern", "flow"):
        from tllm.workgraph import frontend as _frontend

        return getattr(_frontend, name)
    raise AttributeError(f"module 'tllm' has no attribute {name!r}")


__all__ = [
    "BaseConsumer",
    "BeforeNextUse",
    "TllmRuntime",
    "clear_consumers",
    "default_runtime",
    "extern",
    "flow",
    "make_llm",
    "register_consumer",
]
