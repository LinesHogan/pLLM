#!/usr/bin/env python3
"""Formal update-barrier port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortWrite

UpdateBarrierScope = Literal["completed"]


@dataclass(frozen=True)
class UpdateBarrierLocator(Locator):
    scope: UpdateBarrierScope = "completed"

    def __post_init__(self) -> None:
        if self.scope != "completed":
            raise ValueError(f"unsupported update barrier scope: {self.scope}")


class UpdateBarrier:
    KIND = PortKind.UPDATE_BARRIER
    READABLE = False
    WRITABLE = True
    SUPPORTED_PHASES = ("post_generate",)
    BACKING_VLLM_STRUCT = "consumer-produced update readiness barriers after completed generation."

    @staticmethod
    def write(*, scope: UpdateBarrierScope = "completed", mode: str = "append") -> PortWrite:
        return PortWrite(kind=UpdateBarrier.KIND, locator=UpdateBarrierLocator(scope=scope), mode=str(mode))
