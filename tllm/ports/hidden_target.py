#!/usr/bin/env python3
"""Formal hidden-target port for In-Place TTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortRead

HiddenTargetPhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class HiddenTargetLocator(Locator):
    layer: int
    phase: HiddenTargetPhase = "prefill"
    site: str = "block_output"

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise ValueError(f"unsupported hidden target phase: {self.phase}")
        if not self.site:
            raise ValueError("site must be non-empty")


class HiddenTarget:
    KIND = PortKind.HIDDEN_TARGET
    READABLE = True
    WRITABLE = False
    SUPPORTED_PHASES = ("prefill", "decode")
    BACKING_VLLM_STRUCT = "Hidden target activation used by test-time update rules."

    @staticmethod
    def read(*, layer: int, phase: HiddenTargetPhase = "prefill", role: str = "") -> PortRead:
        return PortRead(kind=HiddenTarget.KIND, locator=HiddenTargetLocator(layer=int(layer), phase=phase), role=role)
