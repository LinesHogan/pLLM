#!/usr/bin/env python3
"""Configuration for the OverRIDE-inspired iterative reweighting consumer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OverRIDEConsumerConfig:
    consumer_id: str = "override"
    enabled: bool = True
    hidden_size: int = 0
    vocab_size: int = 0
    rank: int = 16
    num_iterations: int = 2
    lambd: float = 0.8
    learning_rate: float = 1e-4
