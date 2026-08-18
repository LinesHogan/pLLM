#!/usr/bin/env python3
"""Typed consumer plugin declaration shared by builtin and external plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, cast

from tllm.consumers.base import BaseConsumer

ConfigT = TypeVar("ConfigT")
ConsumerT = TypeVar("ConsumerT")


@dataclass(frozen=True)
class ConsumerPluginSpec(Generic[ConfigT, ConsumerT]):
    name: str
    consumer_type: type[ConsumerT]
    config_type: type[ConfigT]
    factory: Callable[[ConfigT], ConsumerT]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("consumer plugin name must be non-empty")

    def build(self, config: ConfigT | None = None) -> ConsumerT:
        resolved = self.config_type() if config is None else config
        if not isinstance(resolved, self.config_type):
            raise TypeError(
                f"consumer plugin `{self.name}` expected config {self.config_type.__name__}, "
                f"got {type(resolved).__name__}"
            )
        consumer = self.factory(resolved)
        if not isinstance(consumer, self.consumer_type) or not isinstance(consumer, BaseConsumer):
            raise TypeError(
                f"consumer plugin `{self.name}` factory returned {type(consumer).__name__}, "
                f"expected {self.consumer_type.__name__}"
            )
        return consumer

    def activate(self, config: ConfigT | None = None, *, primary: bool = False) -> ConsumerT:
        from tllm.consumers.registry import activate_consumer

        consumer = self.build(config)
        if not isinstance(consumer, BaseConsumer):
            raise TypeError(f"consumer plugin `{self.name}` did not build a BaseConsumer")
        return cast(ConsumerT, activate_consumer(consumer, primary=primary))
