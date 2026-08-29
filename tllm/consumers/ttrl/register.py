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
    resolved = config if config is not None else TTRLConsumerConfig()
    if resolved.implementation == "native":
        raise ValueError(
            "TTRL implementation='native' is a workload; build it "
            "with tllm.consumers.ttrl.legacy_workload.build_ttrl_workload and pass it "
            "to make_llm(..., workloads=(...)), not the legacy consumer plugin"
        )
    return PLUGIN.build(config)


def activate(config: TTRLConsumerConfig | None = None, *, primary: bool = False) -> TTRLConsumer:
    return PLUGIN.activate(config, primary=primary)
