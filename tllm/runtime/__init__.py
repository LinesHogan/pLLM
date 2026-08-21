#!/usr/bin/env python3
"""Runtime layer public surface.

Submodules (``residual_runtime``, ``dispatch_plan``, ``scheduling.*`` ...) are
imported directly by their consumers; the package root exports only the
instance-scoped runtime API.
"""

from __future__ import annotations

from tllm.runtime.runtime import TllmRuntime, default_runtime

__all__ = ["TllmRuntime", "default_runtime"]
