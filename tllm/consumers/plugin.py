#!/usr/bin/env python3
"""Typed consumer plugin declaration shared by builtin and external plugins.

Canonical home: ``tllm.runtime.consumer_plugins``. Re-exported here for
existing consumer ``register.py`` modules.
"""

from __future__ import annotations

from tllm.runtime.consumer_plugins import ConfigT, ConsumerPluginSpec, ConsumerT

__all__ = ["ConfigT", "ConsumerPluginSpec", "ConsumerT"]
