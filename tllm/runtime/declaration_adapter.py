"""Typed boundary for user-defined workload declarations and legacy adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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
    instance: object, definition: DeclarationDefinition | None
) -> object | None:
    if definition is None:
        return instance
    if definition.authority is WorkloadAuthority.NATIVE_REGION:
        return None
    dynamic_instance = cast(Any, instance)
    try:
        adapter: Any = dynamic_instance.legacy_consumer
    except AttributeError:
        raise RuntimeError(
            "legacy-adapter workload declaration must provide legacy_consumer"
        ) from None
    consumer: Any = adapter() if callable(adapter) else adapter
    if consumer is None:
        raise RuntimeError(
            "legacy-adapter workload declaration returned no legacy_consumer"
        )
    return cast(object, consumer)


def consumer_spec(consumer: object) -> "ConsumerSpec":
    from tllm.runtime.consumer_plugins import spec_for_consumer  # pyright: ignore[reportUnknownVariableType]

    return cast(
        "ConsumerSpec",
        cast(Any, spec_for_consumer)(consumer),  # pyright: ignore[reportUnknownVariableType]
    )


def register_legacy_consumer(consumer: object) -> None:
    from tllm.runtime import residual_runtime

    cast(
        Any,
        residual_runtime.register_dispatch_consumer,  # pyright: ignore[reportUnknownMemberType]
    )(consumer)


__all__ = [
    "consumer_spec",
    "legacy_consumer",
    "register_legacy_consumer",
    "workload_definition",
]
