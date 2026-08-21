#!/usr/bin/env python3
"""Base consumer interface for hidden producer-consumer framework.

Canonical home: ``tllm.ports.consumer`` (ports layer). This module re-exports
the public names so existing consumer implementations keep importing from
``tllm.consumers.base``.
"""

from __future__ import annotations

from tllm.ports.consumer import BaseConsumer, ConfigT, FusionGroupHandle

__all__ = ["BaseConsumer", "ConfigT", "FusionGroupHandle"]
