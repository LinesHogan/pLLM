#!/usr/bin/env python3
"""Production wiring for the GTTA consumer package."""

from __future__ import annotations

from tllm.consumers.gtta.config import GTTAConsumerConfig
from tllm.consumers.gtta.consumer import GTTAConsumer
from tllm.consumers.plugin import ConsumerPluginSpec

PLUGIN = ConsumerPluginSpec("gtta", GTTAConsumer, GTTAConsumerConfig, GTTAConsumer)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: GTTAConsumerConfig | None = None) -> GTTAConsumer:
    return PLUGIN.build(config)


def activate(config: GTTAConsumerConfig | None = None, *, primary: bool = False) -> GTTAConsumer:
    return PLUGIN.activate(config, primary=primary)
