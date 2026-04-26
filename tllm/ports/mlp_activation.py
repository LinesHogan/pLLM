#!/usr/bin/env python3
"""Formal MLP activation port for In-Place TTT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortRead

MLPActivationPhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class MLPActivationLocator(Locator):
    layer: int
    phase: MLPActivationPhase = "prefill"
    site: str = "mlp_down_input"

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise ValueError(f"unsupported mlp activation phase: {self.phase}")
        if self.site != "mlp_down_input":
            raise ValueError(f"unsupported mlp activation site: {self.site}")


class MLPActivation:
    KIND = PortKind.MLP_ACTIVATION
    READABLE = True
    WRITABLE = False
    SUPPORTED_PHASES = ("prefill", "decode")
    BACKING_VLLM_STRUCT = "MLP activation before down projection."

    @staticmethod
    def read(*, layer: int, phase: MLPActivationPhase = "prefill", role: str = "") -> PortRead:
        return PortRead(kind=MLPActivation.KIND, locator=MLPActivationLocator(layer=int(layer), phase=phase), role=role)
