#!/usr/bin/env python3
"""Production wiring for the TLM consumer package."""

from __future__ import annotations

from tllm.consumers.tlm.config import TLMConsumerConfig
from tllm.consumers.tlm.consumer import TLMConsumer
from tllm.consumers.plugin import ConsumerPluginSpec

PLUGIN = ConsumerPluginSpec("tlm", TLMConsumer, TLMConsumerConfig, TLMConsumer)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: TLMConsumerConfig | None = None) -> TLMConsumer:
    return PLUGIN.build(config)


def activate(config: TLMConsumerConfig | None = None, *, primary: bool = False) -> TLMConsumer:
    return PLUGIN.activate(config, primary=primary)
