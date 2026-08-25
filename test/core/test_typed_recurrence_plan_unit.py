"""Closed generic finite-recurrence JSON contract tests."""

from __future__ import annotations

import json

import pytest
import tllm._core as _core


def _spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "region_id": "test.recurrence",
        "implementation_id": "backend.test:replay@1",
        "row_admission": {
            "allowed_phases": ["prefill"],
            "mixed_batch": "reject",
        },
        "fusion_policy": "max_rows=2;internalized_edges=state_write_read",
        "operations": [
            {
                "operation_id": "first",
                "primitive_key": "backend.test:layer/0@1",
                "implementation_id": "backend.test:replay@1",
                "operands": [{"kind": "seed", "binding_id": "hidden"}],
                "results": ["v0"],
            },
            {
                "operation_id": "second",
                "primitive_key": "backend.test:layer/0@1",
                "implementation_id": "backend.test:replay@1",
                "operands": [{"kind": "region_value", "value_id": "v0"}],
                "results": ["v1"],
            },
        ],
        "dependencies": [
            {"source": "first", "target": "second", "kind": "state_write_read"}
        ],
        "retained_values": ["v1"],
    }


def _wire(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def test_core_compiles_and_validates_closed_typed_recurrence_plan() -> None:
    plan = _core.compile_typed_recurrence_plan(_wire(_spec()))

    assert isinstance(plan, _core.TypedRecurrencePlan)
    assert plan.schema_version == 1
    assert plan.grouping_kind == _core.ExecutionGroupingKind.Wavefront
    assert plan.retained_values == ["v1"]
    assert len(plan.groups) == 1
    group = plan.groups[0]
    assert isinstance(group, _core.TypedExecutionGroup)
    assert group.group_id == 0
    assert group.primitive_key == "backend.test:layer/0@1"
    assert group.release_after == ["v0"]
    assert [operation.operation_id for operation in group.operations] == ["first", "second"]
    assert group.operations[0].operands[0].kind == _core.OperandKind.Seed
    assert group.operations[1].operands[0].kind == _core.OperandKind.RegionValue
    _core.validate_typed_recurrence_plan(
        _core.compile_typed_recurrence_plan_json(_wire(_spec()))
    )


def test_core_rejects_tampered_liveness_and_consumer_specific_row_policy() -> None:
    plan = json.loads(_core.compile_typed_recurrence_plan_json(_wire(_spec())))
    plan["groups"][0]["release_after"] = []
    with pytest.raises(ValueError, match="release_after"):
        _core.validate_typed_recurrence_plan(_wire(plan))

    plan = json.loads(_core.compile_typed_recurrence_plan_json(_wire(_spec())))
    plan["grouping_kind"] = "serial"
    with pytest.raises(ValueError, match="groups or release_after"):
        _core.validate_typed_recurrence_plan(_wire(plan))

    spec = _spec()
    spec["row_admission"] = {
        "allowed_phases": ["prefill"],
        "mixed_batch": "recirculation_only",
    }
    with pytest.raises(ValueError, match="mixed_batch"):
        _core.compile_typed_recurrence_plan(_wire(spec))

    spec = _spec()
    spec["operations"][0]["effects"] = [  # type: ignore[index]
        {"kind": "unknown_effect", "resource_id": "state"}
    ]
    with pytest.raises(ValueError, match="unknown_effect"):
        _core.compile_typed_recurrence_plan(_wire(spec))


def test_core_reorders_contracted_groups_across_a_cross_dependency() -> None:
    spec = _spec()
    operations = spec["operations"]
    assert isinstance(operations, list)
    operations.insert(
        1,
        {
            "operation_id": "independent",
            "primitive_key": "backend.test:other@1",
            "implementation_id": "backend.test:replay@1",
            "operands": [{"kind": "seed", "binding_id": "other"}],
            "results": ["other"],
        },
    )
    dependencies = spec["dependencies"]
    assert isinstance(dependencies, list)
    dependencies.append({"source": "independent", "target": "second", "kind": "control"})

    plan = _core.compile_typed_recurrence_plan(_wire(spec))
    assert [[op.operation_id for op in group.operations] for group in plan.groups] == [
        ["independent"],
        ["first", "second"],
    ]
