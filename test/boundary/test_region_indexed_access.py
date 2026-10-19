"""Indexed native effects retain old reads and enforce the same runtime rights."""

from dataclasses import replace

import pytest
import torch

from pllm.contracts.region_program import (
    AccessMode,
    AccessRange,
    AffineIndex,
    Call,
    Constant,
    Coordinate,
    ExternalResource,
    Function,
    If,
    IndexTerm,
    Program,
    Resource,
    ResourceAccess,
    ScalarType,
    Scan,
    TensorType,
    Value,
    verify,
)
from pllm.execution.region_compilation import (
    analyze_region_program,
    lower_region_bindings,
    lower_region_schedule,
    validate_region_program,
    validate_region_schedule,
)
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import NativePrimitive
from test.boundary.test_region_pages import owner_fixture
from test.boundary.test_region_program import region

X = ScalarType.INDEX

INDEX = AffineIndex(0, (IndexTerm(0),))
CELL = AccessRange(INDEX, AffineIndex(1, (IndexTerm(0),)))
PREFIX = AccessRange(AffineIndex(), INDEX)


def source(tensor):
    read = Function("read", (X,), (tensor,), ("layers",), (ResourceAccess("layers", AccessMode.READ, (CELL,)),))
    write = Function("write", (X,), (), ("layers",), (ResourceAccess("layers", AccessMode.WRITE, (CELL,)),))
    prefix = Function("prefix", (X,), (tensor,), ("layers",), (ResourceAccess("layers", AccessMode.READ, (PREFIX,)),))
    program = Program(
        region(
            (),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Coordinate((), (Value("one", X),), AffineIndex(1)),
                Coordinate((), (Value("two", X),), AffineIndex(2)),
                Call(("zero",), (Value("old", tensor),), read.name),
                Call(("one",), (), write.name),
                Call(("zero",), (), write.name),
                Call(("two",), (Value("sum", tensor),), prefix.name),
                Call(("zero",), (), write.name),
            ),
            ("old", "sum"),
        ),
        (read, write, prefix),
        (Resource("layers", (tensor,), (2,)),),
    )
    return program


def test_disjoint_write_moves_past_read_but_raw_war_waw_keep_original_versions():
    execution, task, rows, _ = owner_fixture(rounds=1)
    (exported,) = task.region._cells.values()
    tensor = exported.declaration.inputs[0]
    program = source(tensor)
    analysis = analyze_region_program(program)
    assert not any(statement.opaque for statement in analysis.statements)
    moved = "{ S0[] -> [1]; S1[] -> [0]; S2[] -> [2]; S3[] -> [3]; S4[] -> [4] }"
    validate_region_schedule(program, moved)
    with pytest.raises(ValueError):
        validate_region_schedule(program, "{ S0[] -> [2]; S1[] -> [0]; S2[] -> [1]; S3[] -> [3]; S4[] -> [4] }")

    def read(inputs, resources):
        return resources.snapshot("layers", (inputs[0],))

    def write(inputs, resources):
        resources.publish("layers", (inputs[0],), (torch.full(tensor.shape, 10.0 + inputs[0]),))
        return ()

    def prefix(inputs, resources):
        return (sum(resources.snapshot("layers", (i,))[0] for i in range(inputs[0])),)

    callbacks = (read, write, prefix)
    cells = {f.name: NativePrimitive(f, callback) for f, callback in zip(program.functions, callbacks, strict=True)}
    task.program = program
    task.region = RegionExecution(
        program, lower_region_schedule(program, moved), lower_region_bindings(program, moved), cells
    )
    final = execution.step(rows)
    old, total = task.result.values
    torch.testing.assert_close(old, torch.ones(tensor.shape), rtol=0, atol=0)
    torch.testing.assert_close(total, torch.full(tensor.shape, 21.0), rtol=0, atol=0)
    assert tuple(ref.version_id for ref in final) == (2, 1)
    assert len(task.context.page_publications) == 3
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0


@pytest.mark.parametrize("fault", ("read_other", "write_readonly", "outside_domain"))
def test_native_cannot_escape_its_declared_interval_or_access_mode(fault):
    execution, task, rows, _ = owner_fixture(rounds=1)
    (exported,) = task.region._cells.values()
    tensor = exported.declaration.inputs[0]
    read = source(tensor).functions[0]
    program = Program(
        region((Value("index", X),), (Call(("index",), (Value("old", tensor),), read.name),), ("old",)),
        (read,),
        (Resource("layers", (tensor,), (2,)),),
    )
    called = []

    def bad(inputs, resources):
        called.append(True)
        if fault == "write_readonly":
            resources.publish("layers", (0,), (torch.zeros(tensor.shape),))
        return resources.snapshot("layers", (1,))

    analysis = analyze_region_program(program, optimize=False)
    assert analysis.statements[0].opaque  # Entry index has no compile-time bound.
    schedule = analysis.reference_schedule
    task.program, task.inputs = program, {"index": 2 if fault == "outside_domain" else 0}
    task.region = RegionExecution(
        program,
        lower_region_schedule(program, schedule),
        lower_region_bindings(program, schedule),
        {read.name: NativePrimitive(read, bad)},
    )
    with pytest.raises(ValueError, match="declared"):
        execution.step(rows)
    assert bool(called) is (fault != "outside_domain")
    assert not task.context.page_publications
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)


def test_external_ranges_keep_alias_boundary_and_both_verifiers_reject_bad_operands():
    tensor = TensorType((2,), "torch.float32")
    program = source(tensor)
    external = replace(program, resources=(ExternalResource("layers", (2,)),))
    assert all(statement.opaque for statement in analyze_region_program(external, optimize=False).statements)
    bad_range = AccessRange(AffineIndex(0, (IndexTerm(1),)), AffineIndex(1))
    read = replace(program.functions[0], accesses=(ResourceAccess("layers", AccessMode.READ, (bad_range,)),))
    invalid = replace(program, functions=(read, *program.functions[1:]))
    for verifier in (verify, validate_region_program):
        with pytest.raises(ValueError):
            verifier(invalid)


@pytest.mark.parametrize("layers,taps", [((1,), (1,)), ((1, 2), (2, 1)), ((0, 2), (2,))])
def test_decoder_binds_request_and_token_once_for_symbolic_kv_access(monkeypatch, layers, taps):
    from pllm import _core
    from pllm.contracts.backend_attachment import CausalModelDomain
    from pllm.execution.decoder_region import DecoderRegion
    from pllm.execution.paged_access import PagedLayer, PagedResourceView, bind_paged_access
    from pllm.execution.region_resources import PagedResourceBinding

    execution, _, rows, refs = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    tensor = TensorType((1, 4), "torch.float32")
    decoder = DecoderRegion(tensor, layers, CausalModelDomain(2, 3, 8), taps=taps)
    calls = []

    def layer(ordinal, positions, hidden, residual):
        calls.append((ordinal, int(positions.item())))
        return hidden + ordinal + positions, hidden.clone()

    def forbidden(*args):
        raise AssertionError("decoder invocation must use its cold plan")

    for name in ("analyze_region_program", "lower_region_bindings", "lower_region_schedule"):
        monkeypatch.setattr(_core, name, forbidden)
    hidden = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    for request, token in ((0, 0), (1, 7), (0, 3)):
        original = replace(rows[0], request_ordinal=request, token_offset=token)
        caches = tuple(
            PagedLayer(str(i), torch.zeros(2, 1, 8, 4), torch.tensor([[0], [1]]), torch.tensor([request * 8 + token]))
            for i in layers
        )
        plan = bind_paged_access((original,), caches)
        kv = PagedResourceBinding(
            PagedResourceView(decoder.source.kv, plan, tuple((i, str(i)) for i in layers)), invocation.context
        )
        result, residual, *observed = decoder.run(
            invocation.context, original, torch.tensor([token]), hidden, layer, kv=kv
        )
        torch.testing.assert_close(result, hidden + sum(layers) + len(layers) * token, rtol=0, atol=0)
        torch.testing.assert_close(residual, hidden + sum(layers[:-1]) + (len(layers) - 1) * token, rtol=0, atol=0)
        for tap, actual in zip(taps, observed, strict=True):
            preceding = layers[: layers.index(tap)]
            torch.testing.assert_close(
                actual, 2 * (hidden + sum(preceding) + len(preceding) * token) + tap + token, rtol=0, atol=0
            )
    assert calls == [(layer, token) for token in (0, 7, 3) for layer in layers]
    for field, value in (("request_ordinal", 2), ("token_offset", 8)):
        with pytest.raises(ValueError, match="size parameter"):
            decoder.run(invocation.context, replace(rows[0], **{field: value}), torch.tensor([0]), hidden, layer, kv=kv)
    assert len(calls) == 3 * len(layers) and not invocation.context.page_publications
    assert tuple(execution.arena.latest(ref.lineage_id) for ref in refs) == refs
    execution.cancel_step(invocation)
    execution.retire(tuple(row.identity for row in rows))


@pytest.mark.parametrize("enabled", (False, True))
def test_conditional_nested_effects_project_to_the_cells_they_can_touch(enabled):
    execution, task, rows, _ = owner_fixture(rounds=1)
    (exported,) = task.region._cells.values()
    tensor = exported.declaration.inputs[0]
    read, write = source(tensor).functions[:2]
    leaf_args = (Value("index", X),)
    write_body = region(leaf_args, (Call(("index",), (), write.name),), ())
    skip_write = region(leaf_args, (), ())
    inner = region(
        (Value("j", X),),
        (
            Constant((), (Value("yes", ScalarType.BOOL),), True),
            If(("yes", "j"), (), write_body, skip_write),
        ),
        (),
    )
    outer = region((Value("i", X),), (Scan((), (), 1, inner),), ())
    branch = region((), (Scan((), (), 2, outer),), ())
    skip = region((), (), ())
    program = Program(
        region(
            (Value("enabled", ScalarType.BOOL),),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Coordinate((), (Value("one", X),), AffineIndex(1)),
                If(("enabled",), (), branch, skip),
                Call(("one",), (Value("other", tensor),), read.name),
                Call(("zero",), (Value("changed", tensor),), read.name),
            ),
            ("other", "changed"),
        ),
        (read, write),
        (Resource("layers", (tensor,), (2,)),),
    )
    analysis = analyze_region_program(program, optimize=False)
    assert not any(statement.opaque for statement in analysis.statements)
    with pytest.raises(ValueError, match="dependency"):
        validate_region_schedule(program, "{ S0[] -> [1]; S1[] -> [2]; S2[] -> [0] }")
    schedule = "{ S0[] -> [1]; S1[] -> [0]; S2[] -> [2] }"
    calls = []

    def read_(inputs, resources):
        calls.append(("read", inputs[0]))
        return resources.snapshot("layers", (inputs[0],))

    def write_(inputs, resources):
        calls.append(("write", inputs[0]))
        resources.publish("layers", (inputs[0],), (torch.full(tensor.shape, 10.0),))
        return ()

    task.program, task.inputs = program, {"enabled": enabled}
    task.region = RegionExecution(
        program,
        lower_region_schedule(program, schedule),
        lower_region_bindings(program, schedule),
        {read.name: NativePrimitive(read, read_), write.name: NativePrimitive(write, write_)},
    )
    final = execution.step(rows)
    assert calls == [("read", 1), *([("write", 0)] * 2 if enabled else []), ("read", 0)]
    for actual, expected in zip(task.result.values, (2.0, 10.0 if enabled else 1.0), strict=True):
        torch.testing.assert_close(actual, torch.full(tensor.shape, expected), rtol=0, atol=0)
    assert tuple(ref.version_id for ref in final) == (2 if enabled else 0, 0)
    assert len(task.context.page_publications) == (2 if enabled else 0)
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
