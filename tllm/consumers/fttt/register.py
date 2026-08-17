#!/usr/bin/env python3
"""Production wiring for the FTTT consumer package."""

from __future__ import annotations

from tllm.consumers.fttt.config import FTTTConsumerConfig
from tllm.consumers.fttt.consumer import FTTTConsumer


def build(config: FTTTConsumerConfig | None = None) -> FTTTConsumer:
    return FTTTConsumer(config or FTTTConsumerConfig())


def activate(config: FTTTConsumerConfig | None = None, *, primary: bool = False) -> FTTTConsumer:
    from tllm.consumers.registry import activate_consumer

    return activate_consumer(build(config), primary=primary)
