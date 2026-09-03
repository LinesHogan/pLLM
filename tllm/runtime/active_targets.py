#!/usr/bin/env python3
"""Shared helpers for deciding whether hook targets are active."""

from __future__ import annotations

from typing import Protocol


class RuntimeWithTargets(Protocol):
    native_workloads_pending: bool


def runtime_has_active_targets(runtime: RuntimeWithTargets) -> bool:
    return bool(runtime.native_workloads_pending)


def runtime_requires_decode_hidden(runtime: RuntimeWithTargets) -> bool:
    return False
