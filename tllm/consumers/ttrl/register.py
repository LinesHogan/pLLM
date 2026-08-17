#!/usr/bin/env python3
"""Production wiring for the TTRL consumer package."""

from __future__ import annotations

from tllm.consumers.ttrl.config import TTRLConsumerConfig
from tllm.consumers.ttrl.consumer import TTRLConsumer

CONSUMER_TYPE = TTRLConsumer
CONFIG_TYPE = TTRLConsumerConfig


def build(config: TTRLConsumerConfig | None = None) -> TTRLConsumer:
    return TTRLConsumer(config or TTRLConsumerConfig())


def activate(config: TTRLConsumerConfig | None = None, *, primary: bool = False) -> TTRLConsumer:
    from tllm.consumers.registry import activate_consumer

    return activate_consumer(build(config), primary=primary)
