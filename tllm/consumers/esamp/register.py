#!/usr/bin/env python3
"""Production wiring for the ESamp consumer package."""

from __future__ import annotations

from tllm.consumers.esamp.config import ESampConsumerConfig
from tllm.consumers.esamp.consumer import ESampConsumer
from tllm.consumers.plugin import ConsumerPluginSpec

PLUGIN = ConsumerPluginSpec("esamp", ESampConsumer, ESampConsumerConfig, ESampConsumer)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: ESampConsumerConfig | None = None) -> ESampConsumer:
    return PLUGIN.build(config)


def activate(config: ESampConsumerConfig | None = None, *, primary: bool = False) -> ESampConsumer:
    return PLUGIN.activate(config, primary=primary)
