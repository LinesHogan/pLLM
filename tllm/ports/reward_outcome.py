#!/usr/bin/env python3
"""Formal reward outcome port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tllm.ports.base import Locator, PortKind, PortWrite

RewardOutcomeScope = Literal["completed"]


@dataclass(frozen=True)
class RewardOutcomeLocator(Locator):
    scope: RewardOutcomeScope = "completed"

    def __post_init__(self) -> None:
        if self.scope != "completed":
            raise ValueError(f"unsupported reward outcome scope: {self.scope}")


class RewardOutcome:
    KIND = PortKind.REWARD_OUTCOME
    READABLE = False
    WRITABLE = True
    SUPPORTED_PHASES = ("post_generate",)
    BACKING_VLLM_STRUCT = (
        "consumer-produced reward, judge, or label outcomes aligned with "
        "completed generation traces."
    )

    @staticmethod
    def write(*, scope: RewardOutcomeScope = "completed", mode: str = "append") -> PortWrite:
        return PortWrite(kind=RewardOutcome.KIND, locator=RewardOutcomeLocator(scope=scope), mode=str(mode))
