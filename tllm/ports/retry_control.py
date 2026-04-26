#!/usr/bin/env python3
"""Formal retry-control port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortWrite

RetryControlScope = Literal["completed"]


@dataclass(frozen=True)
class RetryControlLocator(Locator):
    scope: RetryControlScope = "completed"

    def __post_init__(self) -> None:
        if self.scope != "completed":
            raise ValueError(f"unsupported retry control scope: {self.scope}")


class RetryControl:
    KIND = PortKind.RETRY_CONTROL
    READABLE = False
    WRITABLE = True
    SUPPORTED_PHASES = ("post_generate",)
    BACKING_VLLM_STRUCT = "consumer-produced retry, stop, or accept intents after completed generation."

    @staticmethod
    def write(*, scope: RetryControlScope = "completed", mode: str = "append") -> PortWrite:
        return PortWrite(kind=RetryControl.KIND, locator=RetryControlLocator(scope=scope), mode=str(mode))
