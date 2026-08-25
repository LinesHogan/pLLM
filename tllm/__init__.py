"""Public tLLM API.

The module-level functions are a thin backward-compatible façade over the
process-wide default :class:`tllm.runtime.runtime.TllmRuntime`. New code
should create an explicit runtime instance:

    runtime = TllmRuntime(consumers=[...])
    llm = runtime.make_llm(...)
"""

from __future__ import annotations

from tllm.consumers.base import BaseConsumer
from tllm.runtime.runtime import TllmRuntime, default_runtime
from tllm.workgraph.frontend import BeforeNextUse, extern, flow


def make_llm(*args: object, **kwargs: object) -> object:
    workloads = kwargs.pop("workloads", None)
    runtime_option_names = (
        "workload_memory_budget_bytes",
        "workload_allow_serial_fallback",
        "workload_persistent_state_bytes",
    )
    runtime_options = {
        name: kwargs.pop(name) for name in runtime_option_names if name in kwargs
    }
    if workloads is not None:
        if not isinstance(workloads, (tuple, list)):
            raise TypeError("make_llm.workloads must be a tuple or list")
        memory_budget = runtime_options.get("workload_memory_budget_bytes", 0)
        allow_serial = runtime_options.get("workload_allow_serial_fallback", True)
        persistent_state = runtime_options.get("workload_persistent_state_bytes", 0)
        if not isinstance(memory_budget, int) or isinstance(memory_budget, bool):
            raise TypeError("workload_memory_budget_bytes must be int")
        if not isinstance(allow_serial, bool):
            raise TypeError("workload_allow_serial_fallback must be bool")
        if not isinstance(persistent_state, int) or isinstance(persistent_state, bool):
            raise TypeError("workload_persistent_state_bytes must be int")
        return TllmRuntime(
            workloads=tuple(workloads),
            workload_memory_budget_bytes=memory_budget,
            workload_allow_serial_fallback=allow_serial,
            workload_persistent_state_bytes=persistent_state,
        ).make_llm(*args, **kwargs)
    if runtime_options:
        raise TypeError("workload compiler options require make_llm(workloads=...)")
    return default_runtime().make_llm(*args, **kwargs)


def register_consumer(consumer: BaseConsumer) -> None:
    default_runtime().register_consumer(consumer)


def clear_consumers() -> None:
    default_runtime().clear_consumers()


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
