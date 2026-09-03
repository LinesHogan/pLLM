#!/usr/bin/env python3
"""Common runtime hook helpers."""

from __future__ import annotations

from tllm.backends.protocol import BackendObservableEvent


def dispatch_runtime_event(
    *,
    runtime: object,
    event: BackendObservableEvent,
) -> int:
    return 0
