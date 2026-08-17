#!/usr/bin/env python3
"""Production wiring for the In-Place-TTT consumer package."""

from __future__ import annotations

from tllm.consumers.inplace_ttt.config import InPlaceTTTConsumerConfig
from tllm.consumers.inplace_ttt.consumer import InPlaceTTTConsumer

CONSUMER_TYPE = InPlaceTTTConsumer
CONFIG_TYPE = InPlaceTTTConsumerConfig


def build(config: InPlaceTTTConsumerConfig | None = None) -> InPlaceTTTConsumer:
    return InPlaceTTTConsumer(config or InPlaceTTTConsumerConfig())


def activate(config: InPlaceTTTConsumerConfig | None = None, *, primary: bool = False) -> InPlaceTTTConsumer:
    from tllm.consumers.registry import activate_consumer

    return activate_consumer(build(config), primary=primary)
