#!/usr/bin/env python3
"""Formal module weight view port for fast-weight overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortRead, PortWrite, WriteMode

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
    WRITABLE = True
    SUPPORTED_PHASES = ("prefill", "decode")
    SUPPORTED_WRITE_MODES = (WriteMode.WRAPPER,)
    BACKING_VLLM_STRUCT = "A borrowed view of a module weight such as MLP down_proj.weight."

    @staticmethod
    def read(*, layer: int, module: str, phase: ModuleWeightPhase = "prefill", role: str = "") -> PortRead:
        return PortRead(
            kind=ModuleWeightView.KIND,
            locator=ModuleWeightViewLocator(layer=int(layer), module=module, phase=phase),
            role=role,
        )

    @staticmethod
    def write(
        *,
        layer: int,
        module: str,
        phase: ModuleWeightPhase = "prefill",
        mode: WriteMode = WriteMode.WRAPPER,
        order: int = 0,
    ) -> PortWrite:
        write_mode = WriteMode(str(mode))
        if write_mode not in ModuleWeightView.SUPPORTED_WRITE_MODES:
            supported = ", ".join(m.value for m in ModuleWeightView.SUPPORTED_WRITE_MODES)
            raise ValueError(
                f"unsupported module weight view write mode `{write_mode}`; supported: {supported}"
            )
        return PortWrite(
            kind=ModuleWeightView.KIND,
            locator=ModuleWeightViewLocator(layer=int(layer), module=module, phase=phase),
            mode=write_mode.value,
            order=int(order),
        )
