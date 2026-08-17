#!/usr/bin/env python3
"""Typed extension slots for the runtime state.

Mechanisms that are not part of the core state schema (fast-weight overlay
state, test-time LoRA service handles, ...) register themselves here instead
of adding ``object``-typed fields to ``ResidualRuntimeState``. Access is
runtime-checked: a wrong type fails fast at the ``get`` call site instead of
leaking an opaque ``object`` downstream.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


class ExtensionRegistry:
    def __init__(self) -> None:
        self._slots: dict[str, object] = {}

    def register(self, name: str, state: object) -> None:
        key = str(name).strip()
        if not key:
            raise ValueError("extension name must be non-empty")
        self._slots[key] = state

    def unregister(self, name: str) -> None:
        self._slots.pop(str(name), None)

    def get(self, name: str, expected_type: type[T]) -> T | None:
        value = self._slots.get(str(name))
        if value is None:
            return None
        if not isinstance(value, expected_type):
            raise TypeError(
                f"runtime extension `{name}` has type {type(value).__name__}, "
                f"expected {expected_type.__name__}"
            )
        return value

    def has(self, name: str) -> bool:
        return str(name) in self._slots
