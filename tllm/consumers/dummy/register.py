#!/usr/bin/env python3
"""Production wiring for the dummy consumer package."""

from __future__ import annotations

from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.dummy.consumer import DummyConsumer

CONSUMER_TYPE = DummyConsumer
CONFIG_TYPE = DummyConsumerConfig


def build(config: DummyConsumerConfig | None = None) -> DummyConsumer:
    return DummyConsumer(config or DummyConsumerConfig())


def activate(config: DummyConsumerConfig | None = None, *, primary: bool = False) -> DummyConsumer:
    from tllm.consumers.registry import activate_consumer

    return activate_consumer(build(config), primary=primary)
