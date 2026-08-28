#!/usr/bin/env python3
"""Configuration for the ReFT intervention workload."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReftConsumerConfig:
    consumer_id: str = "reft"
    enabled: bool = True
    # Decoder layers receiving the LoReFT patch (sorted, deduplicated by the
    # signed ActivationPatchSpec).
    layers: tuple[int, ...] = (15,)
    rank: int = 4
    # Prompt-position selector, pyreft's f1+l1 (first/last N prompt tokens).
    first_tokens: int = 0
    last_tokens: int = 2
    # Nominal host-side training rate; carried for specialization identity,
    # consumed by the transformers-side training loop (no in-engine optimizer).
    learning_rate: float = 4e-3
