#!/usr/bin/env python3
"""Consumer plugin mechanism: typed specs, name registry, spec rebuild.

Every consumer package exposes a ``register.py`` with a ``PLUGIN``
(:class:`ConsumerPluginSpec`); this module maps names to those plugins so a
consumer can be instantiated by name — from benchmarks and serving entry
points, from the control plane, and from the cross-process bootstrap, which
rebuilds consumers in the EngineCore child from JSON-safe specs.

Builtin plugins are loaded by module path (no static import: consumers are
leaves above the runtime); external plugins arrive via the
``tllm.consumer_plugins`` entry-point group or ``register_consumer_plugin``.

``tllm.consumers.registry`` / ``tllm.consumers.plugin`` re-export this
surface for existing callers.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Callable, Generic, Protocol, TypeVar, cast

from tllm.common.dataclass_json import dataclass_from_dict, dataclass_to_jsonable
from tllm.ports.consumer import BaseConsumer
from tllm.runtime.mp.bootstrap import ConsumerSpec

ConfigT = TypeVar("ConfigT")
ConsumerT = TypeVar("ConsumerT")


@dataclass(frozen=True)
class ConsumerPluginSpec(Generic[ConfigT, ConsumerT]):
    """Typed consumer plugin declaration shared by builtin and external plugins."""

    name: str
    consumer_type: type[ConsumerT]
    config_type: type[ConfigT]
    factory: Callable[[ConfigT], ConsumerT]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("consumer plugin name must be non-empty")

    def build(self, config: ConfigT | None = None) -> ConsumerT:
        resolved = self.config_type() if config is None else config
        if not isinstance(resolved, self.config_type):
            raise TypeError(
                f"consumer plugin `{self.name}` expected config {self.config_type.__name__}, "
                f"got {type(resolved).__name__}"
            )
        consumer = self.factory(resolved)
        if not isinstance(consumer, self.consumer_type) or not isinstance(consumer, BaseConsumer):
            raise TypeError(
                f"consumer plugin `{self.name}` factory returned {type(consumer).__name__}, "
                f"expected {self.consumer_type.__name__}"
            )
        return consumer

    def activate(self, config: ConfigT | None = None, *, primary: bool = False) -> ConsumerT:
        consumer = self.build(config)
        if not isinstance(consumer, BaseConsumer):
            raise TypeError(f"consumer plugin `{self.name}` did not build a BaseConsumer")
        return cast(ConsumerT, activate_consumer(consumer, primary=primary))


class _PluginModule(Protocol):
    PLUGIN: object


_BUILTIN_PLUGIN_MODULES = {
    name: f"tllm.consumers.{name}.register"
    for name in (
        "esamp",
        "fttt",
        "gtta",
        "inplace_ttt",
        "override",
        "tlm",
        "ttrl",
        "dummy",
    )
}
_CONSUMER_PLUGINS: dict[str, ConsumerPluginSpec[Any, Any]] = {}
_ENTRY_POINTS: dict[str, metadata.EntryPoint] | None = None


def register_consumer_plugin(plugin: object) -> None:
    """Register one typed plugin spec; this is the external plugin boundary."""
    if not isinstance(plugin, ConsumerPluginSpec):
        raise TypeError("consumer plugin must be a ConsumerPluginSpec")
    typed = cast(ConsumerPluginSpec[Any, Any], plugin)
    key = typed.name.strip()
    if not key:
        raise ValueError("consumer plugin name must be non-empty")
    existing = _CONSUMER_PLUGINS.get(key)
    if existing is not None and existing is not typed:
        raise ValueError(f"consumer plugin `{key}` already registered")
    _CONSUMER_PLUGINS[key] = typed


def unregister_consumer_plugin(name: str) -> None:
    """Remove a programmatically registered plugin (primarily test/dev use)."""
    _CONSUMER_PLUGINS.pop(str(name).strip(), None)


def _consumer_entry_points() -> dict[str, metadata.EntryPoint]:
    global _ENTRY_POINTS
    if _ENTRY_POINTS is None:
        discovered = metadata.entry_points()
        selected = discovered.select(group="tllm.consumer_plugins")
        _ENTRY_POINTS = {entry.name: entry for entry in selected}
    return _ENTRY_POINTS


def _load_plugin(name: str) -> ConsumerPluginSpec[Any, Any]:
    key = str(name).strip()
    existing = _CONSUMER_PLUGINS.get(key)
    if existing is not None:
        return existing
    module_name = _BUILTIN_PLUGIN_MODULES.get(key)
    if module_name is not None:
        loaded = cast(_PluginModule, importlib.import_module(module_name)).PLUGIN
    else:
        entry_point = _consumer_entry_points().get(key)
        if entry_point is None:
            known = ", ".join(registered_consumers()) or "<none>"
            raise ValueError(f"unknown consumer `{key}`; registered: {known}")
        loaded = entry_point.load()
    if not isinstance(loaded, ConsumerPluginSpec):
        raise TypeError("consumer plugin must be a ConsumerPluginSpec")
    if loaded.name.strip() != key:
        raise ValueError(
            f"consumer plugin requested as `{key}` declared mismatched name `{loaded.name}`"
        )
    register_consumer_plugin(loaded)
    plugin = _CONSUMER_PLUGINS.get(key)
    if plugin is None:
        raise RuntimeError(f"consumer plugin `{key}` was not retained after registration")
    return plugin


def registered_consumers() -> tuple[str, ...]:
    return tuple(
        sorted(
            set(_BUILTIN_PLUGIN_MODULES)
            | set(_CONSUMER_PLUGINS)
            | set(_consumer_entry_points())
        )
    )


def build_consumer(name: str, config: object | None) -> BaseConsumer:
    consumer = _load_plugin(name).build(config)
    if not isinstance(consumer, BaseConsumer):
        raise TypeError(f"consumer plugin `{name}` did not build a BaseConsumer")
    return consumer


def spec_for_consumer(consumer: BaseConsumer) -> ConsumerSpec:
    """Describe a live consumer as a JSON-safe (name, config) pair.

    The reverse lookup uses the consumer/config types registered alongside
    each builder; the config must be a plain dataclass (see
    ``tllm.common.dataclass_json``).
    """
    for name in registered_consumers():
        plugin = _load_plugin(name)
        if isinstance(consumer, plugin.consumer_type):
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
        plugin = _load_plugin(key)
    except ValueError as exc:
        known = ", ".join(registered_consumers()) or "<none>"
        raise ValueError(f"consumer `{key}` has no registered types; known: {known}") from exc
    return build_consumer(key, dataclass_from_dict(plugin.config_type, spec.config))


def activate_consumer(consumer: BaseConsumer, *, primary: bool = False) -> BaseConsumer:
    """Bind a built consumer to the default runtime.

    ``primary=True`` installs it as THE runtime consumer (ESamp-style);
    otherwise it joins the dispatch registry alongside other consumers.
    """
    from tllm.runtime import residual_runtime
    from tllm.runtime.mp.control import active_control_plane

    plane = active_control_plane()
    spec = spec_for_consumer(consumer) if plane is not None else None

    if primary:
        residual_runtime.clear_dispatch_consumers()
        residual_runtime.set_runtime_consumer(consumer)
    else:
        residual_runtime.register_dispatch_consumer(consumer)
    from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec

    refresh_exported_bootstrap_spec()
    if plane is not None and spec is not None:
        plane.register_consumer(spec.name, spec.config, primary=primary)
    return consumer
