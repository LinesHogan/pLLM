#!/usr/bin/env python3
"""Configuration for the FTTT-inspired feedback edit consumer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FTTTConsumerConfig:
    consumer_id: str = "fttt"
    enabled: bool = True
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3
    max_queue_size: int = 128
    feedback_feature_size: int = 32
    positive_reward_threshold: float = 1.0
    max_retries: int = 0
    edit_scale: float = 1.0
