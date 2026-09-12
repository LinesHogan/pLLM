"""The native verifier checks the same source program, not Python verdicts."""

import json
from dataclasses import replace
from enum import StrEnum

import pytest

from pllm import _core
from pllm.contracts.region_program import (
    AccessMode, Block, BorrowedResource, Call, Constant, Function, InvalidProgram, Program, Publish, Region,
    Resource, ResourceAccess, Scalar, ScalarKind, ScalarType, SizeParameter, Snapshot, TensorType, Value, verify,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.region_compilation import validate_region_program
from test.boundary.test_region_program import grid_program


@pytest.mark.parametrize("chunks, layers", [(0, 3), (1, 1), (2, 5), (6, 4), (1_000_000, 16)])
def test_nested_source_roundtrip_and_native_verification(chunks, layers):
    program = grid_program(chunks, layers)
    wire = encode_boundary(program)
    assert decode_boundary(wire, Program) == program
    validate_region_program(program)
    assert len(wire) < 5000  # Loops stay structural even at large extents.


def resource_program():
    integer = ScalarType.INT
    return Program(Region(Block((), (
        Snapshot((), (Value("x", integer),), "w"),
        Publish(("x",), (), "w"),
    ), ("x",))), resources=(Resource("w", (integer,)),))


def test_native_verifier_accepts_explicit_state_chain():
    program = resource_program()
    assert decode_boundary(encode_boundary(program), Program) == program
    validate_region_program(program)


@pytest.mark.parametrize("change", ["undefined", "missing_payload", "extra_payload", "unknown", "stored_order", "extra_field"])
def test_forged_wire_cannot_bypass_resource_or_closed_schema_rules(change):
    wire = json.loads(encode_boundary(resource_program()))
    block = wire["entry"]["block"]
    if change == "undefined":
        block["operations"][1]["inputs"][0] = "missing"
    elif change == "missing_payload":
        block["operations"][1]["inputs"] = []
    elif change == "extra_payload":
        block["operations"][1]["inputs"].append("x")
    elif change == "unknown":
        block["operations"][1]["resource"] = "not_w"
    elif change == "stored_order":
        wire["resources"][0]["payload"] = [{"kind": "bank", "size": 1, "element": {"kind": "order", "resource": "w"}}]
    else:
        block["operations"][1]["trust_me"] = True
    payload = json.dumps(wire)
    with pytest.raises(ValueError):
        _core.validate_region_program(payload)
    with pytest.raises((ValueError, InvalidProgram)):
        verify(decode_boundary(payload, Program))


def test_native_verifier_checks_region_closure_without_frontend_help():
    wire = json.loads(encode_boundary(grid_program(3, 4)))
    inner = wire["entry"]["block"]["operations"][0]["body"]["block"]["operations"][1]["body"]["block"]
    inner["operations"][1]["inputs"][0] = "hidden"  # Exists only at the program entry.
    with pytest.raises(ValueError, match="captures must be explicit"):
        _core.validate_region_program(json.dumps(wire))


@pytest.mark.parametrize("value, type_", [(True, ScalarType.BOOL), (1.0, ScalarType.FLOAT), (1, ScalarType.INT)])
def test_scalar_literal_types_survive_the_wire(value, type_):
    program = Program(Region(Block((), (Constant((), (Value("x", type_),), value),), ("x",))))
    assert decode_boundary(encode_boundary(program), Program) == program
    validate_region_program(program)


@pytest.mark.parametrize("value", (2**63, 2**80, -(2**63) - 1))
def test_python_integer_literals_cannot_turn_into_floats_at_the_native_boundary(value):
    program = Program(Region(Block((), (Constant((), (Value("x", ScalarType.FLOAT),), value),), ("x",))))
    with pytest.raises(InvalidProgram, match="64-bit"):
        verify(program)


def test_native_unsigned_overflow_is_not_retried_as_a_floating_literal():
    wire = json.loads(encode_boundary(Program(Region(Block((),
        (Constant((), (Value("x", ScalarType.FLOAT),), 1.0),), ("x",))))))
    for literal in (2**63, 2**64 - 1, {"$serde_json::private::Number": "1.0"}):
        wire["entry"]["block"]["operations"][0]["value"] = literal
        with pytest.raises(ValueError):
            _core.validate_region_program(json.dumps(wire))


def test_source_schema_rejects_python_coercions_before_semantic_validation():
    class OtherKind(StrEnum):
        ADD = "add"
    class Integer(int):
        pass
    integer = ScalarType.INT
    valid = Program(Region(Block((Value("x", integer),),
        (Scalar(("x", "x"), (Value("y", integer),), ScalarKind.ADD),), ("y",))))
    operation, = valid.entry.block.operations
    invalid = [replace(valid, entry=Region(replace(valid.entry.block,
        operations=(replace(operation, kind=kind),)))) for kind in ("add", OtherKind.ADD)]
    invalid.extend((
        Program(Region(Block((), (), ())), resources=(Resource("state", [integer]),)),
        Program(Region(Block((Value("x", TensorType((Integer(2),), "torch.float32")),), (), ("x",)))),
        replace(valid, parameters=(SizeParameter("n", Integer(0), 2),)),
    ))
    read = Function("read", (), (), ("borrow",), (ResourceAccess("borrow", "read"),))
    borrowed = Program(Region(Block((), (Call((), (), read.name),), ())),
        (read,), (BorrowedResource("borrow", ()),))
    invalid.append(borrowed)
    for program in invalid:
        with pytest.raises(InvalidProgram):
            verify(program)
    verify(replace(borrowed, functions=(replace(read, accesses=(ResourceAccess("borrow", AccessMode.READ),)),)))
    assert verify(valid) == (integer,)


@pytest.mark.parametrize("change", ["valid", "duplicate", "drop", "stale"])
def test_obsolete_order_sources_cannot_enter_either_planning_boundary(change):
    token = {"kind": "order", "resource": "w"}
    wire = json.loads(encode_boundary(resource_program()))
    block = wire["entry"]["block"]
    block["arguments"] = [{"name": "q0", "type": token}]
    block["operations"][0]["inputs"] = ["q0"]
    block["operations"][0]["outputs"].append({"name": "q1", "type": token})
    block["operations"][1]["inputs"].append("q0" if change == "stale" else "q1")
    block["operations"][1]["outputs"] = [{"name": "q2", "type": token}]
    block["yields"].extend([] if change == "drop" else ["q2", "q2"] if change == "duplicate" else ["q2"])
    payload = json.dumps(wire)
    with pytest.raises(ValueError):
        decode_boundary(payload, Program)
    with pytest.raises(ValueError, match="unknown variant.*order"):
        _core.validate_region_program(payload)
