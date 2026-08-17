#!/usr/bin/env python3
"""Production wiring for the ESamp consumer package."""

from __future__ import annotations

from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.consumer import ESampConsumer


def build(config: ESampConsumerConfig | None = None) -> ESampConsumer:
    return ESampConsumer(config or ESampConsumerConfig())


def activate(config: ESampConsumerConfig | None = None, *, primary: bool = False) -> ESampConsumer:
    from tllm.consumers.registry import activate_consumer

    return activate_consumer(build(config), primary=primary)
