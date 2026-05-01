#!/usr/bin/env python3
"""Shared helpers for deciding whether dispatch-plan targets are active."""

from __future__ import annotations

from typing import Any, Protocol


class ActivityPlan(Protocol):
    def has_active_targets(self) -> bool:
        ...


class RuntimeWithTargets(Protocol):
    dispatch_plan: ActivityPlan | None
    consumer: object | None


def _has_active_sampler_provider(runtime: RuntimeWithTargets) -> bool:
    consumer = getattr(runtime, "consumer", None)
    if consumer is None:
        return False
    getter = getattr(consumer, "sampler_modifier_provider", None)
    if not callable(getter):
        return False
    provider: Any = getter()
    is_active = getattr(provider, "is_active", None)
    return bool(callable(is_active) and is_active())


def runtime_has_active_targets(runtime: RuntimeWithTargets) -> bool:
    plan = runtime.dispatch_plan
    return bool((plan is not None and plan.has_active_targets()) or _has_active_sampler_provider(runtime))
