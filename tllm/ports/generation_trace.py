#!/usr/bin/env python3
"""Formal generation trace port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortRead

GenerationTraceScope = Literal["completed"]


@dataclass(frozen=True)
class GenerationTraceLocator(Locator):
    scope: GenerationTraceScope = "completed"

    def __post_init__(self) -> None:
        if self.scope != "completed":
            raise ValueError(f"unsupported generation trace scope: {self.scope}")


class GenerationTrace:
    KIND = PortKind.GENERATION_TRACE
    READABLE = True
    WRITABLE = False
    SUPPORTED_PHASES = ("post_generate",)
    BACKING_VLLM_STRUCT = (
        "completed vLLM RequestOutput objects after generation, normalized to "
        "request/prompt/sample completion traces."
    )

    @staticmethod
    def read(*, scope: GenerationTraceScope = "completed") -> PortRead:
        return PortRead(kind=GenerationTrace.KIND, locator=GenerationTraceLocator(scope=scope))
