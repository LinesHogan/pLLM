#!/usr/bin/env python3
"""Production wiring for the FTTT consumer package."""

from __future__ import annotations

from tllm.consumers.fttt.config import FTTTConsumerConfig
from tllm.consumers.fttt.consumer import FTTTConsumer
from tllm.consumers.plugin import ConsumerPluginSpec

PLUGIN = ConsumerPluginSpec("fttt", FTTTConsumer, FTTTConsumerConfig, FTTTConsumer)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: FTTTConsumerConfig | None = None) -> FTTTConsumer:
    return PLUGIN.build(config)


def activate(config: FTTTConsumerConfig | None = None, *, primary: bool = False) -> FTTTConsumer:
    return PLUGIN.activate(config, primary=primary)
