#!/usr/bin/env python3
"""Production wiring for the TTRL consumer package."""

from __future__ import annotations

from tllm.consumers.ttrl.config import TTRLConsumerConfig
from tllm.consumers.ttrl.consumer import TTRLConsumer
from tllm.consumers.plugin import ConsumerPluginSpec

PLUGIN = ConsumerPluginSpec("ttrl", TTRLConsumer, TTRLConsumerConfig, TTRLConsumer)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: TTRLConsumerConfig | None = None) -> TTRLConsumer:
    return PLUGIN.build(config)


def activate(config: TTRLConsumerConfig | None = None, *, primary: bool = False) -> TTRLConsumer:
    return PLUGIN.activate(config, primary=primary)
