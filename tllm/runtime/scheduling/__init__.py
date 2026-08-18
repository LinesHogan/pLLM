#!/usr/bin/env python3
"""Shared scheduling infrastructure for off-critical-path consumer work."""

from tllm.runtime.scheduling.deferred import DeferredWork, DeferredWorkQueue

__all__ = ["DeferredWork", "DeferredWorkQueue"]
