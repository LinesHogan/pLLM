#!/usr/bin/env python3
"""Configuration for the TTRL consumer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TTRLConsumerConfig:
    consumer_id: str = "ttrl"
    enabled: bool = True
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3
