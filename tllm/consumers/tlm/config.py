#!/usr/bin/env python3
"""Configuration for the TLM online/offline TTL consumer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TLMMode = Literal["online", "offline"]


@dataclass
class TLMConsumerConfig:
    consumer_id: str = "tlm"
    enabled: bool = True
    mode: TLMMode = "online"
    rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 5e-5
    threshold: float = 3.0
    lamb: float = 0.1
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if self.mode not in ("online", "offline"):
            raise ValueError("TLMConsumerConfig mode must be 'online' or 'offline'")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("TLM rank must be a positive int")
        if (
            isinstance(self.lora_alpha, bool)
            or not isinstance(self.lora_alpha, int)
            or self.lora_alpha < 1
        ):
            raise ValueError("TLM lora_alpha must be a positive int")
        if not 0.0 <= float(self.lora_dropout) < 1.0:
            raise ValueError("TLM lora_dropout must be in [0, 1)")
        if tuple(self.target_modules) != ("q_proj", "v_proj"):
            raise ValueError("TLM currently trains exactly q_proj and v_proj")
        if float(self.learning_rate) <= 0.0:
            raise ValueError("TLM learning_rate must be positive")
        if float(self.lamb) < 0.0:
            raise ValueError("TLM lamb must be non-negative")
