#!/usr/bin/env python3
"""Native TTRL reward-head workload."""

from tllm.consumers.ttrl.rewards import (
    TTRLClosedLoopIntent,
    TTRLMajorityResult,
    TTRLRetryScheduleEntry,
    TTRLRewardResult,
    TTRLSample,
    TTRLSampleReward,
    build_ttrl_closed_loop_intent,
    build_ttrl_reward_outcome_batch,
    build_ttrl_reward_result,
    majority_vote_rewards,
    select_ttrl_training_indices,
)
from tllm.consumers.ttrl.verifier import BoxedNormalizedVerifier, TTRLVerifier
from tllm.consumers.ttrl.workload import TTRLConfig, TTRLWorkload

__all__ = [
    "BoxedNormalizedVerifier",
    "TTRLClosedLoopIntent",
    "TTRLConfig",
    "TTRLMajorityResult",
    "TTRLRetryScheduleEntry",
    "TTRLRewardResult",
    "TTRLSample",
    "TTRLSampleReward",
    "TTRLVerifier",
    "TTRLWorkload",
    "build_ttrl_closed_loop_intent",
    "build_ttrl_reward_outcome_batch",
    "build_ttrl_reward_result",
    "majority_vote_rewards",
    "select_ttrl_training_indices",
]
