"""Opaque backend resources have explicit effects and invocation-scoped endpoints."""

import json
from dataclasses import replace

import pytest

from pllm import _core
from pllm.contracts.region_program import (
    Call,
    ExternalResource,
    Function,
    Program,
    ScalarType,
    Snapshot,
    Value,
    verify,
)
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.region_compilation import analyze_region_program, validate_region_program, validate_region_schedule
from pllm.execution.region_native import external_primitive
from pllm.execution.region_resources import ExternalFunction, RegionResources
from test.boundary.test_region_control import bind
from test.boundary.test_region_pages import owner_fixture
from test.boundary.test_region_program import region
from test.region_reference import interpret

I = ScalarType.INT


def external_program():
    first = Function("owner.first", (), (I,), ("first",))
    second = Function("owner.second", (), (I,), ("second",))
    return Program(
        region(
            (),
            (
                Call((), (Value("x", I),), first.name),
                Call((), (Value("y", I),), second.name),
            ),
            ("x", "y"),
        ),
        (first, second),
        (ExternalResource("first"), ExternalResource("second")),
    )


def test_external_resource_codec_and_verifiers_reject_snapshot_and_unknown_fields():
    program = external_program()
    verify(program)
    validate_region_program(program)
    assert decode_boundary(encode_boundary(program), Program) == program
    for field, value in (("payload", []), ("undeclared", []), ("kind", "unknown")):
        wire = json.loads(encode_boundary(program))
        wire["resources"][0][field] = value
        with pytest.raises(ValueError):
            _core.validate_region_program(json.dumps(wire))
        with pytest.raises((ValueError, KeyError)):
            decode_boundary(json.dumps(wire), Program)
    block = program.entry.block
    invalid = replace(
        program,
        entry=replace(
            program.entry,
            block=replace(block, operations=(Snapshot((), (Value("x", I),), "first"), block.operations[1])),
        ),
    )
    with pytest.raises(ValueError, match="external resources require imported calls"):
        validate_region_program(invalid)


def test_different_external_names_do_not_prove_backend_alias_separation():
    program = external_program()
    analysis = analyze_region_program(program)
    assert all(statement.opaque for statement in analysis.statements)
    with pytest.raises(ValueError):
        validate_region_schedule(program, "{ S0[] -> [1]; S1[] -> [0] }")
    execution, _, rows, refs = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    events = []

    # Both endpoints deliberately share a backend effect; independent resource
    # names must not permit an interchange in the absence of an alias proof.
    def call(inputs):
        events.append(len(events) + 1)
        return (events[-1],)

    functions = {f.name: external_primitive(f) for f in program.functions}
    resources = RegionResources(
        program, invocation.context, (), tuple(ExternalFunction(f, call) for f in program.functions)
    )
    result = bind(program, functions).run({}, pages=resources)
    assert result.values == (1, 2) and events == [1, 2]
    assert not invocation.context.page_publications
    assert tuple(execution.arena.latest(ref.lineage_id) for ref in refs) == refs
    events.clear()
    oracle = interpret(program, {}, functions={f.name: call for f in program.functions})
    assert oracle.values == result.values
    assert tuple(event.resource for event in oracle.effects) == ("first", "second")
    execution.cancel_step(invocation)
    with pytest.raises(RuntimeError, match="released"):
        resources.external(program.functions[0], ())
    execution.retire(tuple(row.identity for row in rows))
