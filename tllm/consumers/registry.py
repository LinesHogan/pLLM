#!/usr/bin/env python3
"""Consumer registry: production wiring for the consumer packages.

Canonical home: ``tllm.runtime.consumer_plugins`` (the plugin mechanism is
runtime infrastructure). This module re-exports the public surface so
existing callers keep importing from ``tllm.consumers.registry``.
"""

from __future__ import annotations

from tllm.runtime.consumer_plugins import (
    activate_consumer,
    build_consumer,
    build_consumer_from_spec,
    describe_consumer_or_none,
    register_consumer_plugin,
    registered_consumers,
    spec_for_consumer,
    unregister_consumer_plugin,
)

__all__ = [
    "activate_consumer",
    "build_consumer",
    "build_consumer_from_spec",
    "describe_consumer_or_none",
    "register_consumer_plugin",
    "registered_consumers",
    "spec_for_consumer",
    "unregister_consumer_plugin",
]
