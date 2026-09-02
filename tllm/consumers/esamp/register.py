#!/usr/bin/env python3
"""Production wiring for the ESamp consumer package."""

from __future__ import annotations

from tllm.consumers.esamp.config import BaseESampConsumer, ESampConsumerConfig
from tllm.consumers.esamp.consumer import ESampConsumer
from tllm.consumers.esamp.program_consumer import ESampProgramConsumer
from tllm.consumers.plugin import ConsumerPluginSpec


def _build_esamp(config: ESampConsumerConfig) -> BaseESampConsumer:
    if str(config.implementation) == "program":
        return ESampProgramConsumer(config)
    if str(config.implementation) == "native":
        raise ValueError(
            "implementation='native' is a native @workload; construct "
            "ESampWorkload(ESampConfig(...)) and pass it to "
            "TllmRuntime(workloads=...)"
        )
    return ESampConsumer(config)


PLUGIN = ConsumerPluginSpec("esamp", BaseESampConsumer, ESampConsumerConfig, _build_esamp)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: ESampConsumerConfig | None = None) -> BaseESampConsumer:
    return PLUGIN.build(config)


def activate(
    config: ESampConsumerConfig | None = None, *, primary: bool = False
) -> BaseESampConsumer:
    return PLUGIN.activate(config, primary=primary)
