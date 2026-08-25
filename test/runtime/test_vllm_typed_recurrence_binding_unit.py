"""Generic vLLM typed-recurrence binding contracts."""

from __future__ import annotations

import json

import pytest
import tllm._core as _core

from tllm.runtime.typed_recurrence import (
    BackendRegionAdapter,
    PrimitiveAdapterRegistry,
    TypedRecurrenceExecutor,
)


def _plan() -> object:
    spec = {
        "schema_version": 1,
        "region_id": "test.region",
        "implementation_id": "test.extern:replay@1",
        "row_admission": {"allowed_phases": ["prefill"], "mixed_batch": "reject"},
        "fusion_policy": "none",
        "operations": [
            {"operation_id": "first", "primitive_key": "vllm.layer_replay", "implementation_id": "test.extern:replay@1", "operands": [{"kind": "seed", "binding_id": "hidden"}, {"kind": "backend_resource", "resource_id": "bias"}], "results": ["v0"]},
            {"operation_id": "second", "primitive_key": "vllm.layer_replay", "implementation_id": "test.extern:replay@1", "operands": [{"kind": "region_value", "value_id": "v0"}, {"kind": "backend_resource", "resource_id": "bias"}], "results": ["answer"]},
        ],
        "dependencies": [{"source": "first", "target": "second", "kind": "data"}],
        "retained_values": ["answer"],
    }
    return _core.compile_typed_recurrence_plan(
        json.dumps(spec, sort_keys=True, separators=(",", ":"))
    )


def test_typed_recurrence_executor_dispatches_only_rust_typed_keys() -> None:
    observed: list[tuple[str, tuple[object, ...]]] = []

    def execute(operation: object, operands: tuple[object, ...]) -> tuple[object, ...]:
        observed.append((operation.operation_id, operands))  # type: ignore[attr-defined]
        return (operands[0] + operands[1],)

    registry = PrimitiveAdapterRegistry((BackendRegionAdapter("test.extern:replay@1", "vllm.layer_replay", execute),))
    values = TypedRecurrenceExecutor(registry).execute(
        _plan(), seed_values={"hidden": 4}, backend_resources={"bias": 3}
    )
    assert observed == [("first", (4, 3)), ("second", (7, 3))]
    assert values == {"answer": 10}


def test_typed_recurrence_executor_rejects_missing_signed_adapter() -> None:
    with pytest.raises(RuntimeError, match="no registered primitive adapter"):
        TypedRecurrenceExecutor(PrimitiveAdapterRegistry(())).execute(
            _plan(), seed_values={"hidden": 1}, backend_resources={"bias": 1}
        )
