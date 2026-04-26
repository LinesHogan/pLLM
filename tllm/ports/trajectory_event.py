#!/usr/bin/env python3
"""Formal trajectory-event port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortWrite

TrajectoryEventScope = Literal["post_step"]


@dataclass(frozen=True)
class TrajectoryEventLocator(Locator):
    scope: TrajectoryEventScope = "post_step"

    def __post_init__(self) -> None:
        if self.scope != "post_step":
            raise ValueError(f"unsupported trajectory event scope: {self.scope}")


class TrajectoryEventPort:
    KIND = PortKind.TRAJECTORY_EVENT
    READABLE = False
    WRITABLE = True
    SUPPORTED_PHASES = ("post_step",)
    BACKING_VLLM_STRUCT = "environment trajectory events produced outside vLLM model execution."

    @staticmethod
    def write(*, scope: TrajectoryEventScope = "post_step", mode: str = "append") -> PortWrite:
        return PortWrite(kind=TrajectoryEventPort.KIND, locator=TrajectoryEventLocator(scope=scope), mode=str(mode))
