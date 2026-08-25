"""Mechanical execution of Rust-signed finite recurrence plans.

Rust owns the recurrence grammar, schedule legality, and liveness. This
runtime module binds only typed operands to exact primitive adapters; it has
no dependency on a backend or a workload.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import tllm._core as _core

PrimitiveExecutor = Callable[[object, tuple[object, ...]], tuple[object, ...]]


def _signed_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Rust typed recurrence {label} must be a non-empty str")
    return value


@dataclass(frozen=True, slots=True)
class PrimitiveAdapter:
    """One exact primitive implementation bound by its signed identity."""

    implementation_id: str
    primitive_key: str
    execute: PrimitiveExecutor

    def __post_init__(self) -> None:
        _signed_text(self.implementation_id, "adapter implementation_id")
        _signed_text(self.primitive_key, "adapter primitive_key")
        if not callable(self.execute):
            raise TypeError("typed recurrence primitive adapter execute must be callable")


# Stable compatibility name while callers migrate to the backend-neutral type.
BackendRegionAdapter = PrimitiveAdapter


class PrimitiveAdapterRegistry:
    """Immutable adapter map indexed only by signed implementation/primitive."""

    __slots__ = ("_adapters",)

    def __init__(self, adapters: tuple[PrimitiveAdapter, ...]) -> None:
        if not isinstance(adapters, tuple) or any(
            not isinstance(adapter, PrimitiveAdapter) for adapter in adapters
        ):
            raise TypeError("primitive adapters must be tuple[PrimitiveAdapter, ...]")
        indexed = {
            (adapter.implementation_id, adapter.primitive_key): adapter
            for adapter in adapters
        }
        if len(indexed) != len(adapters):
            raise ValueError("primitive adapter registry has duplicate signed keys")
        self._adapters = MappingProxyType(indexed)

    def require(self, implementation_id: str, primitive_key: str) -> PrimitiveAdapter:
        key = (
            _signed_text(implementation_id, "group implementation_id"),
            _signed_text(primitive_key, "group primitive_key"),
        )
        return self._require_signed(*key)

    def _require_signed(
        self, implementation_id: str, primitive_key: str
    ) -> PrimitiveAdapter:
        """Look up identifiers already guaranteed by a Rust plan PyClass."""

        try:
            return self._adapters[(implementation_id, primitive_key)]
        except KeyError as exc:
            raise RuntimeError(
                "Rust typed recurrence has no registered primitive adapter for "
                f"implementation={implementation_id!r}, primitive={primitive_key!r}"
            ) from exc


class TypedRecurrenceExecutor:
    """Execute only the dataflow and liveness already signed by Rust."""

    __slots__ = ("_registry",)

    def __init__(self, registry: PrimitiveAdapterRegistry) -> None:
        if not isinstance(registry, PrimitiveAdapterRegistry):
            raise TypeError("typed recurrence executor requires PrimitiveAdapterRegistry")
        self._registry = registry

    @staticmethod
    def _operand_value(
        operand: _core.TypedOperand,
        *,
        values: Mapping[str, object],
        seed_values: Mapping[str, object],
        backend_resources: Mapping[str, object],
    ) -> object:
        if operand.kind == _core.OperandKind.Seed:
            binding_id = operand.binding_id
            source = seed_values
        elif operand.kind == _core.OperandKind.RegionValue:
            binding_id = operand.value_id
            source = values
        elif operand.kind == _core.OperandKind.BackendResource:
            binding_id = operand.resource_id
            source = backend_resources
        else:
            raise RuntimeError("Rust typed recurrence operand has an unknown kind")
        if binding_id is None:
            raise RuntimeError(
                "Rust typed recurrence operand kind has no corresponding signed binding"
            )
        try:
            return source[binding_id]
        except KeyError as exc:
            raise RuntimeError(
                "Rust typed recurrence operand is unavailable: "
                f"kind={operand.kind!r}, binding_id={binding_id!r}"
            ) from exc

    def execute(
        self,
        plan: _core.TypedRecurrencePlan,
        *,
        seed_values: Mapping[str, object],
        backend_resources: Mapping[str, object],
    ) -> Mapping[str, object]:
        if not isinstance(seed_values, Mapping) or not isinstance(
            backend_resources, Mapping
        ):
            raise TypeError("typed recurrence inputs must be mappings")
        if not isinstance(plan, _core.TypedRecurrencePlan):
            raise TypeError("typed recurrence executor requires Rust TypedRecurrencePlan")
        retained = tuple(plan.retained_values)

        values: dict[str, object] = {}
        for group in tuple(plan.groups):
            adapter = self._registry._require_signed(
                group.implementation_id, group.primitive_key
            )
            for operation in tuple(group.operations):
                operation_id = operation.operation_id
                operands = tuple(
                    self._operand_value(
                        operand,
                        values=values,
                        seed_values=seed_values,
                        backend_resources=backend_resources,
                    )
                    for operand in tuple(operation.operands)
                )
                result_ids = tuple(operation.results)
                if any(value_id in values for value_id in result_ids):
                    raise RuntimeError(
                        f"Rust typed recurrence redefines a Region value in {operation_id!r}"
                    )
                outputs = adapter.execute(operation, operands)
                if not isinstance(outputs, tuple) or len(outputs) != len(result_ids):
                    raise RuntimeError(
                        "primitive adapter result arity differs from Rust typed recurrence "
                        f"operation {operation_id!r}"
                    )
                values.update(zip(result_ids, outputs, strict=True))
            for value_id in tuple(group.release_after):
                if value_id not in values:
                    raise RuntimeError(
                        "Rust typed recurrence releases an unavailable Region value "
                        f"{value_id!r}"
                    )
                del values[value_id]
        if set(values) != set(retained):
            raise RuntimeError(
                "Rust typed recurrence signed retained values do not match live Region values"
            )
        return MappingProxyType(dict(values))


__all__ = [
    "BackendRegionAdapter",
    "PrimitiveAdapter",
    "PrimitiveAdapterRegistry",
    "TypedRecurrenceExecutor",
]
