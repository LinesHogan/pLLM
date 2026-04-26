#!/usr/bin/env python3
"""Formal module weight view port for fast-weight overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortRead

ModuleWeightPhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class ModuleWeightViewLocator(Locator):
    layer: int
    module: str
    phase: ModuleWeightPhase = "prefill"

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise ValueError(f"unsupported module weight phase: {self.phase}")
        if not self.module:
            raise ValueError("module must be non-empty")
        if self.module != self.module.strip():
            raise ValueError("module must not contain surrounding whitespace")


class ModuleWeightView:
    KIND = PortKind.MODULE_WEIGHT_VIEW
    READABLE = True
    WRITABLE = False
    SUPPORTED_PHASES = ("prefill", "decode")
    BACKING_VLLM_STRUCT = "A borrowed view of a module weight such as MLP down_proj.weight."

    @staticmethod
    def read(*, layer: int, module: str, phase: ModuleWeightPhase = "prefill", role: str = "") -> PortRead:
        return PortRead(
            kind=ModuleWeightView.KIND,
            locator=ModuleWeightViewLocator(layer=int(layer), module=module, phase=phase),
            role=role,
        )
