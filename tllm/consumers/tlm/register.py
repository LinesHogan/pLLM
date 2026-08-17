#!/usr/bin/env python3
"""Production wiring for the TLM consumer package."""

from __future__ import annotations

from tllm.consumers.tlm.config import TLMConsumerConfig
from tllm.consumers.tlm.consumer import TLMConsumer

CONSUMER_TYPE = TLMConsumer
CONFIG_TYPE = TLMConsumerConfig


def build(config: TLMConsumerConfig | None = None) -> TLMConsumer:
    return TLMConsumer(config or TLMConsumerConfig())


def activate(config: TLMConsumerConfig | None = None, *, primary: bool = False) -> TLMConsumer:
    from tllm.consumers.registry import activate_consumer

    return activate_consumer(build(config), primary=primary)
