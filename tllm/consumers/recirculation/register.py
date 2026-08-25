#!/usr/bin/env python3
"""Production wiring for Recirculation."""

from tllm.consumers.plugin import ConsumerPluginSpec
from tllm.consumers.recirculation.config import RecirculationConfig
from tllm.consumers.recirculation.consumer import RecirculationConsumer

PLUGIN = ConsumerPluginSpec(
    "recirculation", RecirculationConsumer, RecirculationConfig, RecirculationConsumer
)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: RecirculationConfig) -> RecirculationConsumer:
    return PLUGIN.build(config)


def activate(config: RecirculationConfig, *, primary: bool = False) -> RecirculationConsumer:
    return PLUGIN.activate(config, primary=primary)
