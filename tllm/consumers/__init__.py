#!/usr/bin/env python3
"""Consumer interfaces and implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tllm.consumers.base import BaseConsumer

if TYPE_CHECKING:
    from tllm.consumers.esamp import ESampConsumerConfig


def __getattr__(name: str) -> object:
    if name == "ESampConsumerConfig":
        from tllm.consumers.esamp import ESampConsumerConfig

        return ESampConsumerConfig
    raise AttributeError(name)

__all__ = ["BaseConsumer", "ESampConsumerConfig"]
