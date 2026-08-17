#!/usr/bin/env python3
"""Consumer registry: production wiring for the consumer packages.

Every consumer package exposes a ``register.py`` with ``build(config)`` and
``activate(config)``; this registry maps names to those builders so a
consumer can be instantiated by name (benchmarks, serving entry points)
instead of only from tests.
"""

from __future__ import annotations

from typing import Callable

from tllm.common.dataclass_json import dataclass_from_dict, dataclass_to_jsonable
from tllm.consumers.base import BaseConsumer
from tllm.runtime.mp.bootstrap import ConsumerSpec

Builder = Callable[[object], BaseConsumer]

_CONSUMER_BUILDERS: dict[str, Builder] = {}
_BUILDER_TYPES: dict[str, tuple[type[BaseConsumer], type]] = {}


def register_consumer_builder(
    name: str,
    builder: Builder,
    *,
    consumer_type: type[BaseConsumer] | None = None,
    config_type: type | None = None,
) -> None:
    key = str(name).strip()
    if not key:
        raise ValueError("consumer name must be non-empty")
    if key in _CONSUMER_BUILDERS and _CONSUMER_BUILDERS[key] is not builder:
        raise ValueError(f"consumer builder `{key}` already registered")
    _CONSUMER_BUILDERS[key] = builder
    if consumer_type is not None and config_type is not None:
        _BUILDER_TYPES[key] = (consumer_type, config_type)


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


def spec_for_consumer(consumer: BaseConsumer) -> ConsumerSpec:
    """Describe a live consumer as a JSON-safe (name, config) pair.

    The reverse lookup uses the consumer/config types registered alongside
    each builder; the config must be a plain dataclass (see
    ``tllm.common.dataclass_json``).
    """
    for name, (consumer_type, _config_type) in _BUILDER_TYPES.items():
        if isinstance(consumer, consumer_type):
            config = consumer.config
            if config is None or isinstance(config, type):
                raise ValueError(f"consumer `{name}` exposes no config to snapshot")
            return ConsumerSpec(name=name, config=dataclass_to_jsonable(config))
    raise ValueError(f"no registered builder knows consumer type {type(consumer).__name__}")


def describe_consumer_or_none(consumer: BaseConsumer) -> ConsumerSpec | None:
    """Like ``spec_for_consumer`` but returns None for unregistered types.

    Ad-hoc consumer classes (test probes, user experiments) cannot be
    rebuilt from a spec; the bootstrap records them as skipped instead of
    failing the whole capture.
    """
    try:
        return spec_for_consumer(consumer)
    except ValueError:
        return None


def build_consumer_from_spec(spec: ConsumerSpec) -> BaseConsumer:
    """Rebuild a consumer from its spec (child-process bootstrap path)."""
    key = str(spec.name).strip()
    try:
        _consumer_type, config_type = _BUILDER_TYPES[key]
    except KeyError as exc:
        known = ", ".join(sorted(_BUILDER_TYPES)) or "<none>"
        raise ValueError(f"consumer `{key}` has no registered types; known: {known}") from exc
    return build_consumer(key, dataclass_from_dict(config_type, spec.config))


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
        register_consumer_builder(
            name,
            register.build,
            consumer_type=register.CONSUMER_TYPE,
            config_type=register.CONFIG_TYPE,
        )


_register_builtin_builders()
