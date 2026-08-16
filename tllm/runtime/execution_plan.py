#!/usr/bin/env python3
"""Runtime-level execution plan: default producer registry + compilation.

The default registry carries the migrated built-in producers (residual
decode/prefill). ``compile_default_plan`` enforces fail-fast consumer-port
resolution only for the kinds the new registry covers; legacy hardcoded
producers keep serving the remaining kinds until they migrate.
"""

from __future__ import annotations

from typing import Any, Sequence

from tllm.ports.base import PortKind
from tllm.producers.base import HookPoint
from tllm.producers.registry import (
    ExecutionPlan,
    ProducerRegistry,
    compile_execution_plan,
)
from tllm.producers.residual import default_residual_producers

_DEFAULT_REGISTRY: ProducerRegistry | None = None


def get_default_registry() -> ProducerRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = ProducerRegistry()
        for producer in default_residual_producers():
            registry.register(producer)
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY


def enforced_port_kinds(registry: ProducerRegistry) -> frozenset[PortKind]:
    return frozenset(
        spec.kind for entry in registry.entries() for spec in entry.producer.outputs
    )


def compile_default_plan(
    *,
    consumers: Sequence[Any] = (),
    backend_capabilities: Any = None,
    supported_hooks: frozenset[HookPoint] | None = None,
) -> ExecutionPlan:
    registry = get_default_registry()
    return compile_execution_plan(
        registry=registry,
        consumers=consumers,
        backend_capabilities=backend_capabilities,
        supported_hooks=supported_hooks,
        enforced_kinds=enforced_port_kinds(registry),
    )
