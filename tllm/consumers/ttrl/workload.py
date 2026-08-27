#!/usr/bin/env python3
"""Workload dispatcher: select the TTRL implementation from the config."""

from __future__ import annotations

from tllm.consumers.ttrl.config import TTRLConsumerConfig
from tllm.consumers.ttrl.consumer import TTRLConsumer
from tllm.consumers.ttrl.region_workload import TTRLRegionWorkload


def build_ttrl_workload(
    config: TTRLConsumerConfig | None = None,
) -> TTRLConsumer | TTRLRegionWorkload:
    resolved = config if config is not None else TTRLConsumerConfig()
    if resolved.implementation == "region":
        return TTRLRegionWorkload(resolved)
    return TTRLConsumer(resolved)


__all__ = ["build_ttrl_workload"]
