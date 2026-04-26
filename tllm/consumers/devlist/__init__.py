#!/usr/bin/env python3
"""Shared primitives for devlist-inspired consumers."""

from tllm.consumers.devlist.common import LowRankAdapter
from tllm.consumers.devlist.inplace_ttt import (
    InPlaceTTTConsumer,
    InPlaceTTTConsumerConfig,
    InPlaceTTTEngine,
    InPlaceTTTStats,
)
from tllm.consumers.devlist.override import (
    OverRIDEConsumer,
    OverRIDEConsumerConfig,
    OverRIDELogitsResult,
    OverRIDEReweightEngine,
    OverRIDEStats,
)

__all__ = [
    "InPlaceTTTConsumer",
    "InPlaceTTTConsumerConfig",
    "InPlaceTTTEngine",
    "InPlaceTTTStats",
    "LowRankAdapter",
    "OverRIDEConsumer",
    "OverRIDEConsumerConfig",
    "OverRIDELogitsResult",
    "OverRIDEReweightEngine",
    "OverRIDEStats",
]
