"""Native imports keep value versions, declared effects and the original page owner."""

from dataclasses import replace

import pytest
import torch

from pllm.contracts.region_program import (
    AffineIndex, BankType, Call, Coordinate, Function, If, Program, Publish, Resource,
    ScalarType, Scan, Snapshot, TensorType, Value, While,
)
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_native import NativePrimitive, tensor_item
from pllm.execution.region_values import Bank
from pllm.execution.region_reference import interpret
from test.boundary.test_region_control import bind
from test.boundary.test_region_program import region
from test.boundary.test_region_pages import owner_fixture


X, I, B = ScalarType.INDEX, ScalarType.INT, ScalarType.BOOL


@pytest.mark.parametrize("device", ("cpu", "cuda:0"))
@pytest.mark.parametrize("limit,start", ((0, 0.), (2, 0.), (8, 0.), (8, 5.)))
def test_explicit_tensor_predicate_controls_a_pretest_loop(device, limit, start):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    example = torch.zeros((), device=device)
    condition = ExportedCell.capture(lambda x: (x < 3,), ("x",), ("predicate",), (example,))
    update = ExportedCell.capture(lambda x: (x + 1,), ("x",), ("next",), (example,))
    item = tensor_item(condition.declaration.outputs[0])
    t = condition.declaration.inputs[0]
    test = region((Value("x", t),), (
        Call(("x",), (Value("tensor_predicate", condition.declaration.outputs[0]),), condition.digest),
        Call(("tensor_predicate",), (Value("predicate", B),), item.declaration.name),
    ), ("predicate",))
    body = region((Value("x", t),), (Call(("x",), (Value("next", t),), update.digest),), ("next",))
    program = Program(region((Value("x", t),), (
        While(("x",), (Value("final", t),), limit, test, body),
    ), ("final",)), (condition.declaration, update.declaration, item.declaration))
    result = bind(program, {condition.digest: condition, update.digest: update, item.declaration.name: item}).run(
        {"x": torch.full((), start, device=device)})
    assert result.values[0].item() == start + min(limit, max(0, 3 - start))


@pytest.mark.parametrize("dtype,value,result_type", (("torch.bool", True, B), ("torch.int64", -(2**63), I),
    ("torch.float64", 1.25, ScalarType.FLOAT)))
def test_tensor_scalar_import_preserves_python_scalar_type(dtype, value, result_type):
    type_ = TensorType((), dtype)
    primitive = tensor_item(type_)
    program = Program(region((Value("x", type_),), (
        Call(("x",), (Value("y", result_type),), primitive.declaration.name),
    ), ("y",)), (primitive.declaration,))
    result = bind(program, {primitive.declaration.name: primitive}).run(
        {"x": torch.tensor(value, dtype=getattr(torch, dtype.removeprefix("torch.")))})
    assert result.values == (value,)
    assert type(result.values[0]) is type(value)


def test_native_mutation_cannot_overwrite_ssa_inputs_or_leave_borrowed_outputs():
    type_ = TensorType((3,), "torch.float32")
    buffer = torch.zeros(3)
    saved = []
    def mutate(inputs, resources):
        x, = inputs
        x.add_(3)
        buffer.copy_(x)
        saved.append(resources)
        return (buffer,)
    declaration = Function("native.test.update", (type_,), (type_,))
    primitive = NativePrimitive(declaration, mutate)
    program = Program(region((Value("x", type_),), (
        Call(("x",), (Value("y", type_),), declaration.name),
    ), ("x", "y")), (declaration,))
    source = torch.ones(3)
    result = bind(program, {declaration.name: primitive}).run({"x": source})
    buffer.zero_()
    torch.testing.assert_close(result.values[0], source, rtol=0, atol=0)
    torch.testing.assert_close(result.values[1], torch.full((3,), 4.), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="expired"):
        saved[0].snapshot("undeclared")


def native_page_program(tensor, count, dynamic):
    function = Function("native.test.page_update", (X,), (tensor,), ("layers",))
    body = region((Value("i", X),), (
        Coordinate((), (Value("zero", X),), AffineIndex(0)),
        Call(("zero",), (Value("old", tensor),), function.name),
    ), ("old",))
    outputs = (Value("old_values", BankType(tensor, count)),)
    branch = region((), (Scan((), outputs, count, body),), ("old_values",))
    entry = region((Value("enabled", B),), (
        If(("enabled",), outputs, branch, branch),
    ), ("old_values",)) if dynamic else branch
    return Program(entry, (function,), (Resource("layers", (tensor,), (2,)),))


@pytest.mark.parametrize("dynamic", (False, True))
@pytest.mark.parametrize("cuda", (False, True))
def test_native_effects_in_scan_use_exact_original_versions_and_owned_stream(dynamic, cuda):
    if cuda and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    stream = torch.cuda.Stream() if cuda else None
    execution, task, rows, _ = owner_fixture(rounds=1, stream=stream)
    exported, = task.region._cells.values()
    program = native_page_program(exported.declaration.inputs[0], 4, dynamic)
    declaration, = program.functions
    views = []
    def update(inputs, resources):
        if stream is not None:
            assert torch.cuda.current_stream() == stream
        index, = inputs
        old, = resources.snapshot("layers", (index,))
        resources.publish("layers", (index,), (old + 2,))
        views.append(resources)
        return (old,)
    primitive = NativePrimitive(declaration, update)
    task.program, task.region = program, bind(program, {declaration.name: primitive})
    task.inputs = {"enabled": True} if dynamic else {}
    def oracle(inputs):
        index, family = inputs
        old = family.elements[index]
        return old, Bank((old + 2, family.elements[1]))
    t = exported.declaration.inputs[0]
    expected = interpret(program, task.inputs, functions={declaration.name: oracle},
        resources={"layers": (Bank((torch.ones(t.shape, device=t.device), torch.full(t.shape, 2., device=t.device))),)})
    final = execution.step(rows, stream)
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    for actual, want in zip(task.result.values[0].elements, expected.values[0].elements, strict=True):
        torch.testing.assert_close(actual, want, rtol=0, atol=0)
    assert tuple(ref.version_id for ref in final) == (4, 0)
    assert len(task.context.page_publications) == 4
    for view in views:
        with pytest.raises(RuntimeError, match="expired"):
            view.publish("layers", (0,), (torch.zeros(t.shape, device=t.device),))
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0


@pytest.mark.parametrize("fault", ("undeclared", "boolean_index", "bad_result", "cancel", "raise"))
def test_native_failure_expires_scope_and_preserves_original_commit_boundary(fault):
    from pllm.execution.state_execution import StateSettlementError
    execution, task, rows, refs = owner_fixture(rounds=1)
    exported, = task.region._cells.values()
    program = native_page_program(exported.declaration.inputs[0], 3, False)
    declaration, = program.functions
    views = []
    def fail(inputs, resources):
        views.append(resources)
        if fault == "undeclared":
            resources.snapshot("other", (0,))
        if fault == "boolean_index":
            resources.snapshot("layers", (False,))
        old, = resources.snapshot("layers", (0,))
        resources.publish("layers", (0,), (old + 1,))
        if fault == "bad_result":
            return (7,)
        if fault == "cancel":
            task.context.cancel_requested.set()
            return (old,)
        raise RuntimeError("native failure after commit")
    task.program = program
    task.region = bind(program, {declaration.name: NativePrimitive(declaration, fail)})
    invocation = execution.start_step(rows)
    committed = fault not in ("undeclared", "boolean_index")
    with pytest.raises(StateSettlementError if committed else ValueError):
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert len(views) == 1
    with pytest.raises(RuntimeError, match="expired"):
        views[0].snapshot("layers", (0,))
    assert execution.arena.latest(refs[0].lineage_id).version_id == int(committed)
    if committed:
        execution.cancel_step(invocation)
    else:
        with pytest.raises(RuntimeError, match="not registered as active"):
            execution.cancel_step(invocation)
    assert not invocation.context.page_writes
    assert all(release is not None for release in invocation.context.page_pins.values())
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0


def test_native_signature_must_match_the_source_import():
    declaration = Function("native.test", (I,), (I,))
    program = Program(region((Value("n", I),), (
        Call(("n",), (Value("out", I),), declaration.name),
    ), ("out",)), (declaration,))
    with pytest.raises(ValueError, match="source declaration"):
        bind(program, {declaration.name: NativePrimitive(replace(declaration, inputs=(B,)), lambda x, r: x)})
    class UndeclaredCallable:
        def __init__(self):
            self.declaration = declaration
        def invoke(self, inputs, pages):
            raise AssertionError("unrecognized callable was executed")
    with pytest.raises(ValueError, match="explicit native import"):
        bind(program, {declaration.name: UndeclaredCallable()})
