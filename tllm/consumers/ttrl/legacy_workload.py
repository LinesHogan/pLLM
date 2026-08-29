#!/usr/bin/env python3
"""Legacy TTRL migration dispatcher."""

from __future__ import annotations

from tllm.consumers.ttrl.config import TTRLConsumerConfig
from tllm.consumers.ttrl.consumer import TTRLConsumer
from tllm.consumers.ttrl.workload import TTRLConfig, TTRLWorkload


def build_ttrl_workload(
    config: TTRLConsumerConfig | None = None,
) -> TTRLConsumer | TTRLWorkload:
    resolved = config if config is not None else TTRLConsumerConfig()
    if resolved.implementation == "native":
        return TTRLWorkload(
            TTRLConfig(
                consumer_id=resolved.consumer_id,
                enabled=resolved.enabled,
                hidden_size=resolved.hidden_size,
                rank=resolved.rank,
                learning_rate=resolved.learning_rate,
            )
        )
    return TTRLConsumer(resolved)


__all__ = ["build_ttrl_workload"]
