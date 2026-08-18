#!/usr/bin/env python3
"""Production wiring for the In-Place-TTT consumer package."""

from __future__ import annotations

from tllm.consumers.inplace_ttt.config import InPlaceTTTConsumerConfig
from tllm.consumers.inplace_ttt.consumer import InPlaceTTTConsumer
from tllm.consumers.plugin import ConsumerPluginSpec

PLUGIN = ConsumerPluginSpec(
    "inplace_ttt", InPlaceTTTConsumer, InPlaceTTTConsumerConfig, InPlaceTTTConsumer
)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: InPlaceTTTConsumerConfig | None = None) -> InPlaceTTTConsumer:
    return PLUGIN.build(config)


def activate(config: InPlaceTTTConsumerConfig | None = None, *, primary: bool = False) -> InPlaceTTTConsumer:
    return PLUGIN.activate(config, primary=primary)
