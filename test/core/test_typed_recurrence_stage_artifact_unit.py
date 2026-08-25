"""Closed immutable stage artifacts for finite typed recurrences."""

from __future__ import annotations

import json

import pytest
import tllm._core as _core


def _wire(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _spec() -> dict[str, object]:
    return {
        "schema_version": 1,
        "shape_specialization": [
            ["stage/request_count", "1"],
            ["stage/row_count", "2"],
        ],
        "row_count": 2,
        "request_count": 1,
        "recurrence": {
            "schema_version": 1,
            "region_id": "test.typed-stage",
            "implementation_id": "test.backend:stage@1",
            "row_admission": {
                "allowed_phases": ["prefill"],
                "mixed_batch": "reject",
            },
            "fusion_policy": "none",
            "operations": [
                {
                    "operation_id": "first",
                    "primitive_key": "test.backend:first@1",
                    "implementation_id": "test.backend:stage@1",
                    "operands": [
                        {"kind": "seed", "binding_id": "seed/0"},
                        {"kind": "backend_resource", "resource_id": "call/0"},
                    ],
                    "results": ["v0"],
                },
                {
                    "operation_id": "second",
                    "primitive_key": "test.backend:second@1",
                    "implementation_id": "test.backend:stage@1",
                    "operands": [
                        {"kind": "region_value", "value_id": "v0"},
                        {"kind": "backend_resource", "resource_id": "call/1"},
                    ],
                    "results": ["answer"],
                },
            ],
            "dependencies": [{"source": "first", "target": "second", "kind": "data"}],
            "retained_values": ["answer"],
        },
        "seed_bindings": [
            {"binding_id": "seed/0", "kind": "test.seed", "coordinates": [0]},
        ],
        "resource_bindings": [
            {"resource_id": "call/0", "kind": "test.call", "coordinates": [0]},
            {"resource_id": "call/1", "kind": "test.call", "coordinates": [1]},
        ],
        "output_bindings": [
            {
                "output_id": "result/0",
                "value_id": "answer",
                "kind": "test.result",
                "coordinates": [0],
            },
        ],
    }


def _facts(*, phase: str = "prefill") -> str:
    return _wire(
        {
            "schema_version": 1,
            "shape_specialization": [
                ["stage/request_count", "1"],
                ["stage/row_count", "2"],
            ],
            "row_count": 2,
            "request_count": 1,
            "request_spans": [
                {
                    "request_ordinal": 0,
                    "packed_row_start": 0,
                    "packed_row_end": 2,
                    "token_offset_start": 4,
                    "token_offset_end": 6,
                    "phase": phase,
                }
            ],
        }
    )


def _artifact_for_rows(row_count: int) -> str:
    spec = _spec()
    spec["row_count"] = row_count
    spec["shape_specialization"] = [
        ["stage/request_count", "1"],
        ["stage/row_count", str(row_count)],
    ]
    return _core.compile_typed_recurrence_stage_artifact_json(_wire(spec))


def _bundle_for_rows(row_count: int, *, phase: str = "prefill") -> object:
    artifacts = []
    for current_rows in range(1, row_count + 1):
        spec = _spec()
        spec["row_count"] = current_rows
        spec["shape_specialization"] = [
            ["stage/request_count", "1"],
            ["stage/row_count", str(current_rows)],
        ]
        spec["recurrence"]["row_admission"]["allowed_phases"] = [phase]
        artifacts.append(
            {
                "row_count": current_rows,
                "artifact_json": _core.compile_typed_recurrence_stage_artifact_json(
                    _wire(spec)
                ),
            }
        )
    wire = _core.compile_typed_recurrence_stage_artifact_bundle_json(
        _wire({"schema_version": 1, "artifacts": artifacts})
    )
    return _core.load_typed_recurrence_stage_artifact_bundle(wire)


def test_rust_compiles_loads_and_binds_an_immutable_typed_stage_artifact() -> None:
    artifact = _core.compile_typed_recurrence_stage_artifact(_wire(_spec()))

    assert isinstance(artifact, _core.TypedRecurrenceStageArtifact)
    assert len(artifact.signature) == 64
    assert artifact.plan.retained_values == ["answer"]
    assert artifact.resource_bindings[1].coordinates == [1]
    wire = _core.compile_typed_recurrence_stage_artifact_json(_wire(_spec()))
    _core.validate_typed_recurrence_stage_artifact(wire)

    loaded = _core.load_typed_recurrence_stage_artifact(wire)
    bound = _core.bind_typed_recurrence_stage(loaded, _facts())
    assert isinstance(bound, _core.TypedBoundRecurrenceStage)
    assert bound.signature == artifact.signature
    assert bound.plan.groups[0].operations[0].operation_id == "first"
    assert bound.seed_bindings[0].binding_id == "seed/0"


def test_stage_binding_rejects_shape_or_phase_misses_without_replanning() -> None:
    artifact = _core.compile_typed_recurrence_stage_artifact(_wire(_spec()))

    with pytest.raises(ValueError, match="shape_specialization"):
        _core.bind_typed_recurrence_stage(
            artifact,
            _wire(
                {
                    "schema_version": 1,
                    "shape_specialization": [["stage/row_count", "3"]],
                    "row_count": 3,
                    "request_count": 1,
                    "request_spans": [
                        {
                            "request_ordinal": 0,
                            "packed_row_start": 0,
                            "packed_row_end": 3,
                            "token_offset_start": 4,
                            "token_offset_end": 7,
                            "phase": "prefill",
                        }
                    ],
                }
            ),
        )
    with pytest.raises(ValueError, match="row admission"):
        _core.bind_typed_recurrence_stage(artifact, _facts(phase="decode"))


def test_stage_artifact_validation_rejects_tampered_groups_or_resource_coordinates() -> None:
    artifact = json.loads(
        _core.compile_typed_recurrence_stage_artifact_json(_wire(_spec()))
    )
    artifact["plan"]["groups"][0]["release_after"] = ["v0"]
    with pytest.raises(ValueError, match="typed recurrence stage artifact"):
        _core.validate_typed_recurrence_stage_artifact(_wire(artifact))

    artifact = json.loads(
        _core.compile_typed_recurrence_stage_artifact_json(_wire(_spec()))
    )
    artifact["resource_bindings"][0]["coordinates"] = [99]
    with pytest.raises(ValueError, match="signature"):
        _core.validate_typed_recurrence_stage_artifact(_wire(artifact))


def test_stage_artifact_requires_exact_generic_shape_and_coordinate_contracts() -> None:
    spec = _spec()
    spec["shape_specialization"] = [["stage/request_count", "1"]]
    with pytest.raises(ValueError, match="stage/row_count"):
        _core.compile_typed_recurrence_stage_artifact(_wire(spec))

    spec = _spec()
    spec["shape_specialization"] = [
        ["stage/request_count", "1"],
        ["stage/row_count", "3"],
    ]
    with pytest.raises(ValueError, match="stage/row_count"):
        _core.compile_typed_recurrence_stage_artifact(_wire(spec))

    spec = _spec()
    spec["seed_bindings"][0]["coordinates"] = [-1]
    with pytest.raises(ValueError, match="seed bindings"):
        _core.compile_typed_recurrence_stage_artifact(_wire(spec))

    spec = _spec()
    spec["resource_bindings"][0]["coordinates"] = [-1]
    with pytest.raises(ValueError, match="resource bindings"):
        _core.compile_typed_recurrence_stage_artifact(_wire(spec))

    spec = _spec()
    spec["output_bindings"][0]["coordinates"] = [-1]
    with pytest.raises(ValueError, match="output bindings"):
        _core.compile_typed_recurrence_stage_artifact(_wire(spec))

    spec = _spec()
    spec["output_bindings"].append(
        {
            "output_id": "result/1",
            "value_id": "answer",
            "kind": "test.result",
            "coordinates": [1],
        }
    )
    with pytest.raises(ValueError, match="output bindings"):
        _core.compile_typed_recurrence_stage_artifact(_wire(spec))

    spec = _spec()
    spec["recurrence"]["row_admission"]["mixed_batch"] = "noop"
    with pytest.raises(ValueError, match="mixed_batch=noop"):
        _core.compile_typed_recurrence_stage_artifact(_wire(spec))


def test_stage_binding_requires_canonical_request_span_evidence() -> None:
    artifact = _core.compile_typed_recurrence_stage_artifact(_wire(_spec()))

    facts = json.loads(_facts())
    facts["request_spans"][0]["request_ordinal"] = 1
    with pytest.raises(ValueError, match="request_spans"):
        _core.bind_typed_recurrence_stage(artifact, _wire(facts))

    facts = json.loads(_facts())
    facts["request_spans"][0]["packed_row_end"] = 1
    with pytest.raises(ValueError, match="request_spans"):
        _core.bind_typed_recurrence_stage(artifact, _wire(facts))

    facts = json.loads(_facts())
    facts["request_spans"][0]["token_offset_end"] = 7
    with pytest.raises(ValueError, match="request_spans"):
        _core.bind_typed_recurrence_stage(artifact, _wire(facts))

    facts = json.loads(_facts())
    facts["request_spans"] = []
    with pytest.raises(ValueError, match="one canonical request span"):
        _core.bind_typed_recurrence_stage(artifact, _wire(facts))


def test_rust_loads_signed_contiguous_stage_artifact_bundles() -> None:
    bundle_spec = {
        "schema_version": 1,
        "artifacts": [
            {"row_count": 1, "artifact_json": _artifact_for_rows(1)},
            {"row_count": 2, "artifact_json": _artifact_for_rows(2)},
        ],
    }
    wire = _core.compile_typed_recurrence_stage_artifact_bundle_json(
        _wire(bundle_spec)
    )
    _core.validate_typed_recurrence_stage_artifact_bundle(wire)
    bundle = _core.load_typed_recurrence_stage_artifact_bundle(wire)

    assert len(bundle.signature) == 64
    assert [entry.row_count for entry in bundle.artifacts] == [1, 2]
    assert _core.typed_recurrence_stage_artifact_for_rows(bundle, 2).row_count == 2
    with pytest.raises(ValueError, match="no exact row specialization"):
        _core.typed_recurrence_stage_artifact_for_rows(bundle, 3)


def test_stage_artifact_bundle_rejects_noncanonical_rows_or_tampering() -> None:
    bundle_spec = {
        "schema_version": 1,
        "artifacts": [{"row_count": 2, "artifact_json": _artifact_for_rows(2)}],
    }
    with pytest.raises(ValueError, match="cover 1 through N"):
        _core.compile_typed_recurrence_stage_artifact_bundle_json(_wire(bundle_spec))

    bundle_spec = {
        "schema_version": 1,
        "artifacts": [{"row_count": 1, "artifact_json": _artifact_for_rows(1)}],
    }
    wire = json.loads(
        _core.compile_typed_recurrence_stage_artifact_bundle_json(_wire(bundle_spec))
    )
    wire["signature"] = "0" * 64
    with pytest.raises(ValueError, match="bundle signature"):
        _core.load_typed_recurrence_stage_artifact_bundle(_wire(wire))


def test_rust_binds_bundle_directly_from_observed_packed_span() -> None:
    bundle = _bundle_for_rows(3)
    row_plan = _core.plan_packed_prefill_rows([3], [8], [4], 3)

    assert (
        _core.admit_typed_recurrence_stage_bundle_for_packed_plan(bundle, row_plan)
        == _core.TypedStagePlanAdmission.Execute
    )

    bound = _core.bind_typed_recurrence_stage_bundle_for_packed_span(
        bundle, row_plan, 0
    )

    assert bound.stage.signature == bundle.artifacts[2].artifact.signature
    assert bound.observed_span.request_ordinal == 0
    assert bound.observed_span.packed_row_start == 0
    assert bound.observed_span.packed_row_end == 3
    assert bound.observed_span.token_offset_start == 4
    assert bound.observed_span.token_offset_end == 7
    assert bound.observed_span.phase == _core.SchedulerBatchPhase.Prefill

    request_bound = _core.bind_typed_recurrence_stage_bundle_for_request_rows(
        bundle, 3
    )
    assert request_bound.stage.signature == bound.stage.signature
    assert request_bound.observed_span.phase == _core.SchedulerBatchPhase.Prefill


def test_observed_packed_span_binding_derives_phase_and_fails_closed() -> None:
    prefill_bundle = _bundle_for_rows(2)
    decode_plan = _core.plan_packed_prefill_rows([2], [2], [2], 2)
    assert (
        _core.admit_typed_recurrence_stage_bundle_for_packed_plan(
            prefill_bundle, decode_plan
        )
        == _core.TypedStagePlanAdmission.Noop
    )
    with pytest.raises(ValueError, match="row admission"):
        _core.bind_typed_recurrence_stage_bundle_for_packed_span(
            prefill_bundle, decode_plan, 0
        )

    mixed_plan = _core.plan_packed_prefill_rows([2], [2], [1], 2)
    with pytest.raises(ValueError, match="mixed backend phases"):
        _core.admit_typed_recurrence_stage_bundle_for_packed_plan(
            prefill_bundle, mixed_plan
        )
    with pytest.raises(ValueError, match="mixed prefill/decode"):
        _core.bind_typed_recurrence_stage_bundle_for_packed_span(
            prefill_bundle, mixed_plan, 0
        )

    with pytest.raises(ValueError, match="no exact row specialization"):
        _core.bind_typed_recurrence_stage_bundle_for_packed_span(
            _bundle_for_rows(1), decode_plan, 0
        )

    decode_bundle = _bundle_for_rows(2, phase="decode")
    bound = _core.bind_typed_recurrence_stage_bundle_for_packed_span(
        decode_bundle, decode_plan, 0
    )
    assert bound.observed_span.phase == _core.SchedulerBatchPhase.Decode
