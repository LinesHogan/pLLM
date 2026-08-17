#!/usr/bin/env python3
"""In-Place-TTT consumer configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class InPlaceTTTConsumerConfig:
    consumer_id: str = "inplace_ttt"
    enabled: bool = True
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3
    chunk_size: int = 4096
    source_layer: int = 0
    target_layer: int = -1
    phases: tuple[Literal["prefill", "decode"], ...] = ("prefill",)

    def __post_init__(self) -> None:
        normalized: list[Literal["prefill", "decode"]] = []
        for phase in self.phases:
            if phase not in ("prefill", "decode"):
                raise ValueError("InPlaceTTTConsumerConfig phases must contain only 'prefill' or 'decode'")
            if phase not in normalized:
                normalized.append(phase)
        if not normalized:
            raise ValueError("InPlaceTTTConsumerConfig requires at least one phase")
        self.phases = tuple(normalized)


