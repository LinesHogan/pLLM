#!/usr/bin/env python3
"""Shared primitives for devlist-inspired consumers."""

from tllm.consumers.devlist.common import LowRankAdapter
from tllm.consumers.devlist.fttt import (
    FTTTConsumer,
    FTTTConsumerConfig,
    FTTTEngine,
    FTTTInstanceState,
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
    OverRIDEBatchLogitsResult,
    OverRIDEConsumer,
    OverRIDEConsumerConfig,
    OverRIDELogitsResult,
    OverRIDEReweightEngine,
    OverRIDEStats,
)
from tllm.consumers.devlist.ttrl import (
    TTRLMajorityResult,
    TTRLRewardResult,
    TTRLSampleReward,
    TTRLConsumer,
    TTRLConsumerConfig,
    TTRLEngine,
    TTRLSample,
    TTRLStats,
    build_ttrl_reward_result,
    majority_vote_rewards,
    select_ttrl_training_indices,
)
from tllm.consumers.devlist.tlm import (
    TLMConsumer,
    TLMConsumerConfig,
    TLMEngine,
    TLMStats,
    run_tlm_ordered_step,
    select_scalar_score_fallback_samples,
    select_ttl_samples,
)

__all__ = [
    "FTTTConsumer",
    "FTTTConsumerConfig",
    "FTTTEngine",
    "FTTTInstanceState",
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
    "OverRIDEBatchLogitsResult",
    "OverRIDEConsumer",
    "OverRIDEConsumerConfig",
    "OverRIDELogitsResult",
    "OverRIDEReweightEngine",
    "OverRIDEStats",
    "TTRLConsumer",
    "TTRLConsumerConfig",
    "TTRLEngine",
    "TTRLMajorityResult",
    "TTRLRewardResult",
    "TTRLSample",
    "TTRLSampleReward",
    "TTRLStats",
    "TLMConsumer",
    "TLMConsumerConfig",
    "TLMEngine",
    "TLMStats",
    "build_ttrl_reward_result",
    "majority_vote_rewards",
    "parse_reward",
    "run_tlm_ordered_step",
    "select_scalar_score_fallback_samples",
    "select_ttl_samples",
    "select_ttrl_training_indices",
]
