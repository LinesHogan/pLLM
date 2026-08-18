#!/usr/bin/env python3
"""Consumer interfaces and implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tllm.consumers.base import BaseConsumer

if TYPE_CHECKING:
    from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig


def __getattr__(name: str) -> object:
    if name in {"ESampConsumer", "ESampConsumerConfig"}:
        from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig

        return {
            "ESampConsumer": ESampConsumer,
            "ESampConsumerConfig": ESampConsumerConfig,
        }[name]
    raise AttributeError(name)

__all__ = ["BaseConsumer", "ESampConsumer", "ESampConsumerConfig"]
