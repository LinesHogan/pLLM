#!/usr/bin/env python3
"""Compatibility package for devlist-inspired consumers."""

from tllm.consumers._common.alignment import LowRankAdapter
from tllm.consumers.fttt import (
    FTTTConsumer,
    FTTTConsumerConfig,
    FTTTEngine,
    FTTTInstanceState,
    FTTTStats,
    FeedbackRecord,
)
from tllm.consumers.gtta import (
    GTTAConsumer,
    GTTAConsumerConfig,
    GTTAStats,
    TrajectoryFeedback,
    parse_reward,
)
from tllm.consumers.inplace_ttt import (
    InPlaceTTTConsumer,
    InPlaceTTTConsumerConfig,
    InPlaceTTTEngine,
    InPlaceTTTStats,
)
from tllm.consumers.override import (
    OverRIDEBatchLogitsResult,
    OverRIDEConsumer,
    OverRIDEConsumerConfig,
    OverRIDELogitsResult,
    OverRIDEReweightEngine,
    OverRIDEStats,
)
from tllm.consumers.ttrl import (
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
from tllm.consumers.tlm import (
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
