#!/usr/bin/env python3
"""Consumer registry and dispatch plan owner."""

from __future__ import annotations

from typing import Iterable, List

from tllm.ports.consumer import BaseConsumer


class ConsumerRegistry:
    def __init__(self) -> None:
        self._consumers: List[BaseConsumer[object]] = []

    def register(self, consumer: BaseConsumer[object]) -> None:
        self._consumers.append(consumer)

    def clear(self) -> None:
        self._consumers.clear()

    def consumers(self) -> List[BaseConsumer[object]]:
        return list(self._consumers)

    def replace_all(self, consumers: Iterable[BaseConsumer[object]]) -> None:
        self._consumers = list(consumers)
