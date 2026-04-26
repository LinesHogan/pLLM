#!/usr/bin/env python3
"""TTRL consumer package."""

from tllm.consumers.ttrl.consumer import (
    TTRLConsumer,
    TTRLConsumerConfig,
    TTRLEngine,
    TTRLMajorityResult,
    TTRLRewardResult,
    TTRLSample,
    TTRLSampleReward,
    TTRLStats,
    build_ttrl_reward_outcome_batch,
    build_ttrl_reward_result,
    majority_vote_rewards,
    select_ttrl_training_indices,
)

__all__ = [
    "TTRLConsumer",
    "TTRLConsumerConfig",
    "TTRLEngine",
    "TTRLMajorityResult",
    "TTRLRewardResult",
    "TTRLSample",
    "TTRLSampleReward",
    "TTRLStats",
    "build_ttrl_reward_outcome_batch",
    "build_ttrl_reward_result",
    "majority_vote_rewards",
    "select_ttrl_training_indices",
]
