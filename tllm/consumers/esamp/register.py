#!/usr/bin/env python3
"""Production wiring for the ESamp consumer package."""

from __future__ import annotations

from tllm.consumers.esamp.config import BaseESampConsumer, ESampConsumerConfig
from tllm.consumers.esamp.consumer import ESampConsumer
from tllm.consumers.esamp.program_consumer import ESampProgramConsumer
from tllm.consumers.plugin import ConsumerPluginSpec
from tllm.ports.consumer import BaseConsumer


def _build_esamp(config: ESampConsumerConfig) -> BaseConsumer:
    if str(config.implementation) == "program":
        return ESampProgramConsumer(config)
    if str(config.implementation) == "region":
        raise ValueError(
            "implementation='native' is a native @workload; use "
            "build_esamp_workload(config) with TllmRuntime(workloads=...)"
        )
    return ESampConsumer(config)


PLUGIN = ConsumerPluginSpec("esamp", BaseESampConsumer, ESampConsumerConfig, _build_esamp)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: ESampConsumerConfig | None = None) -> BaseConsumer:
    return PLUGIN.build(config)


def activate(config: ESampConsumerConfig | None = None, *, primary: bool = False) -> BaseConsumer:
    return PLUGIN.activate(config, primary=primary)
