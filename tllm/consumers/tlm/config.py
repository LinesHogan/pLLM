#!/usr/bin/env python3
"""Configuration for the TLM-inspired online/offline TTL consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TLMMode = Literal["online", "offline"]


@dataclass
class TLMConsumerConfig:
    consumer_id: str = "tlm"
    enabled: bool = True
    mode: TLMMode = "online"
    hidden_size: int = 0
    rank: int = 16
    learning_rate: float = 1e-3
    threshold: float = 0.5
    lamb: float = 1.0
    adapter_slots: int = 1
    vocab_size: int = 0
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if self.mode not in ("online", "offline"):
            raise ValueError("TLMConsumerConfig mode must be 'online' or 'offline'")
        self.adapter_slots = max(1, int(self.adapter_slots))
