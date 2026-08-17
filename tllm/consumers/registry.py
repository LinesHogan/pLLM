#!/usr/bin/env python3
"""Consumer registry: production wiring for the consumer packages.

Every consumer package exposes a ``register.py`` with ``build(config)`` and
``activate(config)``; this registry maps names to those builders so a
consumer can be instantiated by name (benchmarks, serving entry points)
instead of only from tests.
"""

from __future__ import annotations

from typing import Callable

from tllm.consumers.base import BaseConsumer

Builder = Callable[[object], BaseConsumer]

_CONSUMER_BUILDERS: dict[str, Builder] = {}


def register_consumer_builder(name: str, builder: Builder) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("consumer name must be non-empty")
    if key in _CONSUMER_BUILDERS and _CONSUMER_BUILDERS[key] is not builder:
        raise ValueError(f"consumer builder `{key}` already registered")
    _CONSUMER_BUILDERS[key] = builder


def registered_consumers() -> tuple[str, ...]:
    return tuple(sorted(_CONSUMER_BUILDERS))


def build_consumer(name: str, config: object) -> BaseConsumer:
    key = str(name).strip()
    try:
        builder = _CONSUMER_BUILDERS[key]
    except KeyError as exc:
        known = ", ".join(registered_consumers()) or "<none>"
        raise ValueError(f"unknown consumer `{key}`; registered: {known}") from exc
    return builder(config)


def activate_consumer(consumer: BaseConsumer, *, primary: bool = False) -> BaseConsumer:
    """Bind a built consumer to the default runtime.

    ``primary=True`` installs it as THE runtime consumer (ESamp-style);
    otherwise it joins the dispatch registry alongside other consumers.
    """
    from tllm.runtime import residual_runtime

    if primary:
        residual_runtime.clear_dispatch_consumers()
        residual_runtime.set_runtime_consumer(consumer)
    else:
        residual_runtime.register_dispatch_consumer(consumer)
    return consumer


_BUILTIN_CONSUMERS = ("esamp", "fttt", "gtta", "inplace_ttt", "override", "tlm", "ttrl", "dummy")


def _register_builtin_builders() -> None:
    import importlib

    for name in _BUILTIN_CONSUMERS:
        register = importlib.import_module(f"tllm.consumers.{name}.register")
        register_consumer_builder(name, register.build)


_register_builtin_builders()
