#!/usr/bin/env python3
"""Shared helpers for deciding whether dispatch-plan targets are active."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider
    from tllm.runtime.workload_execution import NativeWorkloadExecution


class ActivityPlan(Protocol):
    def has_active_targets(self) -> bool:
        ...


class SamplerProviderOwner(Protocol):
    def sampler_modifier_provider(self) -> "SamplerModifierProvider | None":
        ...


class RuntimeWithTargets(Protocol):
    dispatch_plan: ActivityPlan | None
    consumer: SamplerProviderOwner | None
    workload_execution: "NativeWorkloadExecution | None"
    native_workloads_pending: bool


def _has_active_sampler_provider(runtime: RuntimeWithTargets) -> bool:
    provider = _sampler_provider(runtime)
    if provider is None:
        return False
    return bool(provider.is_active())


def _sampler_provider(
    runtime: RuntimeWithTargets,
) -> "SamplerModifierProvider | None":
    execution = runtime.workload_execution
    if execution is not None:
        provider = execution.sampler_modifier_provider()
        if provider is not None:
            return provider
    consumer = runtime.consumer
    if consumer is None:
        return None
    return consumer.sampler_modifier_provider()


def runtime_has_active_targets(runtime: RuntimeWithTargets) -> bool:
    plan = runtime.dispatch_plan
    return bool(
        (plan is not None and plan.has_active_targets())
        or _has_active_sampler_provider(runtime)
        or runtime.workload_execution is not None
        or runtime.native_workloads_pending
    )


def runtime_requires_decode_hidden(runtime: RuntimeWithTargets) -> bool:
    """Read the decode-residual requirement projected from the signed Regions."""

    execution = runtime.workload_execution
    return execution is not None and execution.requires_decode_hidden
