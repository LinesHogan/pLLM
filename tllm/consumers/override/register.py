#!/usr/bin/env python3
"""Production wiring for the OverRIDE consumer package."""

from __future__ import annotations

from tllm.consumers.override.config import OverRIDEConsumerConfig
from tllm.consumers.override.consumer import OverRIDEConsumer

CONSUMER_TYPE = OverRIDEConsumer
CONFIG_TYPE = OverRIDEConsumerConfig


def build(config: OverRIDEConsumerConfig | None = None) -> OverRIDEConsumer:
    return OverRIDEConsumer(config or OverRIDEConsumerConfig())


def activate(config: OverRIDEConsumerConfig | None = None, *, primary: bool = False) -> OverRIDEConsumer:
    from tllm.consumers.registry import activate_consumer

    return activate_consumer(build(config), primary=primary)
