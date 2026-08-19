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
    return default_runtime().make_llm(*args, **kwargs)


def register_consumer(consumer: BaseConsumer) -> None:
    default_runtime().register_consumer(consumer)


def clear_consumers() -> None:
    default_runtime().clear_consumers()


__all__ = [
    "BeforeNextUse",
    "TllmRuntime",
    "clear_consumers",
    "default_runtime",
    "extern",
    "flow",
    "make_llm",
    "register_consumer",
]
