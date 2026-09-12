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


@pytest.mark.parametrize("change", ["undefined", "missing_payload", "extra_payload", "unknown", "extra_field"])
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
