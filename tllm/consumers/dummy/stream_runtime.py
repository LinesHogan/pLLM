#!/usr/bin/env python3
"""Optional stream runtime for dummy consumer template."""

from __future__ import annotations

from typing import Callable, Optional

import torch


class DummyStreamRuntime:
    """Lazy side-stream runner.

    The stream is created on first use, not at construction: constructing a
    CUDA stream initializes the CUDA context, and doing that in the parent
    process poisons fork-based EngineCore children under
    VLLM_ENABLE_V1_MULTIPROCESSING=1.
    """

    def __init__(self, enable_async: bool) -> None:
        self._enable_async = bool(enable_async)
        self._stream: Optional[torch.cuda.Stream] = None

    def _ensure_stream(self) -> Optional[torch.cuda.Stream]:
        if self._stream is None and self._enable_async and torch.cuda.is_available():
            self._stream = torch.cuda.Stream()
        return self._stream

    def run(self, fn: Callable[[], None]) -> None:
        stream = self._ensure_stream()
        if stream is None:
            fn()
            return
        with torch.cuda.stream(stream):
            fn()

    def synchronize(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
