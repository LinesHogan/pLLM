#!/usr/bin/env python3
"""Production wiring for the OverRIDE consumer package."""

from __future__ import annotations

from tllm.consumers.override.config import OverRIDEConsumerConfig
from tllm.consumers.override.consumer import OverRIDEConsumer
from tllm.consumers.plugin import ConsumerPluginSpec


def _build_override(config: OverRIDEConsumerConfig) -> OverRIDEConsumer:
    if str(config.implementation) == "region":
        raise ValueError(
            "implementation='region' is a native @workload; use "
            "build_override_workload(config) with TllmRuntime(workloads=...)"
        )
    return OverRIDEConsumer(config)


PLUGIN = ConsumerPluginSpec(
    "override", OverRIDEConsumer, OverRIDEConsumerConfig, _build_override
)
CONSUMER_TYPE = PLUGIN.consumer_type
CONFIG_TYPE = PLUGIN.config_type


def build(config: OverRIDEConsumerConfig | None = None) -> OverRIDEConsumer:
    return PLUGIN.build(config)


def activate(config: OverRIDEConsumerConfig | None = None, *, primary: bool = False) -> OverRIDEConsumer:
    return PLUGIN.activate(config, primary=primary)
