"""Typed boundary for user-defined workload declarations and legacy adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tllm.ports.consumer import BaseConsumer

from tllm.workload import (
    NativeWorkloadDefinition,
    WorkloadAuthority,
    WorkloadDefinition,
    definition_for_workload_class,
)
DeclarationDefinition = WorkloadDefinition | NativeWorkloadDefinition

if TYPE_CHECKING:
    from tllm.runtime.consumer_plugins import ConsumerSpec


def workload_definition(instance: object) -> DeclarationDefinition | None:
    return definition_for_workload_class(type(instance))


def legacy_consumer(
    instance: object, definition: DeclarationDefinition
) -> BaseConsumer[object] | None:
    if definition.authority is WorkloadAuthority.NATIVE_REGION:
        return None
    return definition.legacy_consumer.build(instance)


def consumer_spec(consumer: BaseConsumer[object]) -> "ConsumerSpec":
    from tllm.runtime.consumer_plugins import spec_for_consumer

    return spec_for_consumer(consumer)


def register_legacy_consumer(consumer: BaseConsumer[object]) -> None:
    from tllm.runtime.residual_runtime import register_dispatch_consumer

    register_dispatch_consumer(consumer)


__all__ = [
    "consumer_spec",
    "legacy_consumer",
    "register_legacy_consumer",
    "workload_definition",
]
