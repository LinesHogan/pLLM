#!/usr/bin/env python3
"""TTRL consumer package."""

from tllm.consumers.ttrl.consumer import (
    TTRLConsumer,
    TTRLClosedLoopIntent,
    TTRLConsumerConfig,
    TTRLEngine,
    TTRLMajorityResult,
    TTRLRewardResult,
    TTRLSample,
    TTRLSampleReward,
    TTRLStats,
    build_ttrl_closed_loop_intent,
    build_ttrl_reward_outcome_batch,
    build_ttrl_reward_result,
    majority_vote_rewards,
    select_ttrl_training_indices,
)

__all__ = [
    "TTRLConsumer",
    "TTRLClosedLoopIntent",
    "TTRLConsumerConfig",
    "TTRLEngine",
    "TTRLMajorityResult",
    "TTRLRewardResult",
    "TTRLSample",
    "TTRLSampleReward",
    "TTRLStats",
    "build_ttrl_closed_loop_intent",
    "build_ttrl_reward_outcome_batch",
    "build_ttrl_reward_result",
    "majority_vote_rewards",
    "select_ttrl_training_indices",
]
