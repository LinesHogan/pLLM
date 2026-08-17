#!/usr/bin/env python3
"""Formal residual stream port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import (
    Locator,
    PortKind,
    PortRead,
    PortWrite,
    WriteMode,
    WriteModeName,
    WriteRows,
    WriteRowSelection,
)

ResidualPhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class ResidualLocator(Locator):
    layer: int
    site: str
    phase: ResidualPhase

    def __post_init__(self) -> None:
        if not isinstance(self.layer, int):
            raise ValueError("ResidualLocator.layer must be an int")
        if self.site not in ResidualStream.SUPPORTED_SITES:
            raise ValueError(f"unsupported residual stream site: {self.site}")
        if self.phase not in ResidualStream.SUPPORTED_PHASES:
            raise ValueError(f"unsupported residual stream phase: {self.phase}")


class ResidualStream:
    KIND = PortKind.RESIDUAL_STREAM
    READABLE = True
    WRITABLE = True
    SUPPORTED_PHASES = ("prefill", "decode")
    SUPPORTED_SITES = (
        "block_input",
        "attn_input",
        "attn_output",
        "mlp_input",
        "block_output",
    )
    SUPPORTED_WRITE_MODES = (WriteMode.REPLACE, WriteMode.ADD)
    BACKING_VLLM_STRUCT = (
        "per-layer residual hidden views captured from runtime tap points and "
        "mapped onto logical layer/site locators."
    )

    @staticmethod
    def read(*, layer: int, site: str, phase: ResidualPhase, role: str = "") -> PortRead:
        return PortRead(
            kind=ResidualStream.KIND,
            locator=ResidualLocator(layer=layer, site=site, phase=phase),
            role=role,
        )

    @staticmethod
    def write(
        *,
        layer: int,
        site: str,
        phase: ResidualPhase,
        mode: WriteModeName = WriteMode.REPLACE,
        rows: WriteRowSelection = WriteRows.DECODE_ONLY,
        order: int = 0,
    ) -> PortWrite:
        write_mode = WriteMode(str(mode))
        if write_mode not in ResidualStream.SUPPORTED_WRITE_MODES:
            supported = ", ".join(m.value for m in ResidualStream.SUPPORTED_WRITE_MODES)
            raise ValueError(
                f"unsupported residual stream write mode `{write_mode}`; supported: {supported}"
            )
        return PortWrite(
            kind=ResidualStream.KIND,
            locator=ResidualLocator(layer=layer, site=site, phase=phase),
            mode=write_mode.value,
            rows=rows,
            order=int(order),
        )
