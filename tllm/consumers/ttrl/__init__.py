#!/usr/bin/env python3
"""TTRL consumer package."""

from tllm.consumers.ttrl.config import TTRLConsumerConfig
from tllm.consumers.ttrl.consumer import TTRLConsumer, TTRLStats, run_ttrl_toy_closed_loop
from tllm.consumers.ttrl.program import TTRLProgram
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
    "TTRLConsumer",
    "TTRLClosedLoopIntent",
    "TTRLConsumerConfig",
    "TTRLConfig",
    "TTRLMajorityResult",
    "TTRLProgram",
    "TTRLRetryScheduleEntry",
    "TTRLRewardResult",
    "TTRLSample",
    "TTRLSampleReward",
    "TTRLStats",
    "TTRLVerifier",
    "TTRLWorkload",
    "build_ttrl_closed_loop_intent",
    "build_ttrl_reward_outcome_batch",
    "build_ttrl_reward_result",
    "majority_vote_rewards",
    "run_ttrl_toy_closed_loop",
    "select_ttrl_training_indices",
]
