#!/usr/bin/env python3
"""Production wiring for the GTTA consumer package."""

from __future__ import annotations

from tllm.consumers.gtta.config import GTTAConsumerConfig
from tllm.consumers.gtta.consumer import GTTAConsumer

CONSUMER_TYPE = GTTAConsumer
CONFIG_TYPE = GTTAConsumerConfig


def build(config: GTTAConsumerConfig | None = None) -> GTTAConsumer:
    return GTTAConsumer(config or GTTAConsumerConfig())


def activate(config: GTTAConsumerConfig | None = None, *, primary: bool = False) -> GTTAConsumer:
    from tllm.consumers.registry import activate_consumer

    return activate_consumer(build(config), primary=primary)
