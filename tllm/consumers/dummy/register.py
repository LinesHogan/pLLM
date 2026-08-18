#!/usr/bin/env python3
"""Production wiring for the dummy consumer package."""

from __future__ import annotations

from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.dummy.consumer import DummyConsumer
from tllm.consumers.plugin import ConsumerPluginSpec

PLUGIN = ConsumerPluginSpec("dummy", DummyConsumer, DummyConsumerConfig, DummyConsumer)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: DummyConsumerConfig | None = None) -> DummyConsumer:
    return PLUGIN.build(config)


def activate(config: DummyConsumerConfig | None = None, *, primary: bool = False) -> DummyConsumer:
    return PLUGIN.activate(config, primary=primary)
