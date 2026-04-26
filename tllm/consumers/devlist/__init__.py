#!/usr/bin/env python3
"""Shared primitives for devlist-inspired consumers."""

from tllm.consumers.devlist.common import LowRankAdapter
from tllm.consumers.devlist.fttt import (
    FTTTConsumer,
    FTTTConsumerConfig,
    FTTTEngine,
    FTTTStats,
    FeedbackRecord,
)
from tllm.consumers.devlist.gtta import (
    GTTAConsumer,
    GTTAConsumerConfig,
    GTTAStats,
    TrajectoryFeedback,
    parse_reward,
)
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
from tllm.consumers.devlist.ttrl import (
    TTRLConsumer,
    TTRLConsumerConfig,
    TTRLEngine,
    TTRLSample,
    TTRLStats,
    majority_vote_rewards,
)
from tllm.consumers.devlist.tlm import (
    TLMConsumer,
    TLMConsumerConfig,
    TLMEngine,
    TLMStats,
    select_ttl_samples,
)

__all__ = [
    "FTTTConsumer",
    "FTTTConsumerConfig",
    "FTTTEngine",
    "FTTTStats",
    "FeedbackRecord",
    "GTTAConsumer",
    "GTTAConsumerConfig",
    "GTTAStats",
    "TrajectoryFeedback",
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
    "TTRLConsumer",
    "TTRLConsumerConfig",
    "TTRLEngine",
    "TTRLSample",
    "TTRLStats",
    "TLMConsumer",
    "TLMConsumerConfig",
    "TLMEngine",
    "TLMStats",
    "majority_vote_rewards",
    "parse_reward",
    "select_ttl_samples",
]
