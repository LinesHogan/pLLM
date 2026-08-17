#!/usr/bin/env python3
"""TTRL consumer package."""

from tllm.consumers.ttrl.config import TTRLConsumerConfig
from tllm.consumers.ttrl.consumer import TTRLConsumer, run_ttrl_toy_closed_loop
from tllm.consumers.ttrl.engine import (
    TTRLClosedLoopIntent,
    TTRLEngine,
    TTRLGRPOTrainerBridgeUnavailable,
    TTRLMajorityResult,
    TTRLRetryScheduleEntry,
    TTRLRewardResult,
    TTRLSample,
    TTRLSampleReward,
    TTRLStats,
    build_ttrl_closed_loop_intent,
    build_ttrl_grpo_training_batch,
    build_ttrl_reward_outcome_batch,
    build_ttrl_reward_result,
    create_optional_verl_grpo_bridge,
    majority_vote_rewards,
    select_ttrl_training_indices,
    ttrl_grpo_clipped_ratio_loss,
    ttrl_grpo_group_normalized_advantages,
)
from tllm.consumers.ttrl.verifier import BoxedNormalizedVerifier, TTRLVerifier

__all__ = [
    "BoxedNormalizedVerifier",
    "TTRLConsumer",
    "TTRLClosedLoopIntent",
    "TTRLConsumerConfig",
    "TTRLEngine",
    "TTRLGRPOTrainerBridgeUnavailable",
    "TTRLMajorityResult",
    "TTRLRetryScheduleEntry",
    "TTRLRewardResult",
    "TTRLSample",
    "TTRLSampleReward",
    "TTRLStats",
    "TTRLVerifier",
    "build_ttrl_closed_loop_intent",
    "build_ttrl_grpo_training_batch",
    "build_ttrl_reward_outcome_batch",
    "build_ttrl_reward_result",
    "create_optional_verl_grpo_bridge",
    "majority_vote_rewards",
    "run_ttrl_toy_closed_loop",
    "select_ttrl_training_indices",
    "ttrl_grpo_clipped_ratio_loss",
    "ttrl_grpo_group_normalized_advantages",
]
