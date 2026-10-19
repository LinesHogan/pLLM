"""Region effects on PTTS state: the request owner's leases, COW pages and commits.

Programs snapshot and publish request pages through ``RegionResources``. Every
scenario runs through ``StateExecution`` and checks the user-visible PTTS
contract: published versions, pins, atomic payloads, failure/cancel commit
boundaries and full reclamation after retirement. The multistate scenarios
share one arena between request, route and instance resources.
"""

from dataclasses import dataclass, replace

import pytest
import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_program import (
    AccessMode,
    AccessRange,
    AffineIndex,
    BankType,
    Call,
    Constant,
    Coordinate,
    Function,
    If,
    IndexTerm,
    Program,
    Publish,
    Resource,
    ResourceAccess,
    Scalar,
    ScalarKind,
    ScalarType,
    Scan,
    SizeParameter,
    Snapshot,
    Value,
    While,
)
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.semantic_graph import AnchorId, SemanticState, StateResource, StateScope
from pllm.contracts.tensor_layout import LayoutFieldId, StateTensorLayout, TensorLayoutRequest
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_native import NativePrimitive
from pllm.execution.region_resources import RegionResources, ResourcePageCell
from pllm.execution.region_values import Bank
from pllm.execution.state_execution import MathBinding, StateSettlementError
from pllm.execution.tensor_program import FunctionMath
from test.region_fixtures import (
    bind,
    owner_fixture,
    page_arena,
    page_program,
    region,
    replan_states,
    retire_all,
)
from test.region_reference import interpret

X, I, B = ScalarType.INDEX, ScalarType.INT, ScalarType.BOOL


def assert_same(actual, expected) -> None:
    if isinstance(expected, Bank):
        for a, e in zip(actual.elements, expected.elements, strict=True):
            assert_same(a, e)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        assert actual == expected


def initial_layers(shape, device="cpu"):
    """The reference view of request a's pages: layer 0 holds 1, layer 1 holds 2."""
    return {"layers": (Bank((torch.ones(shape, device=device), torch.full(shape, 2.0, device=device))),)}


@pytest.mark.parametrize("cuda", (False, True))
def test_rescheduled_region_publishes_every_version_through_the_request_owner(cuda):
    if cuda and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    stream = torch.cuda.Stream() if cuda else None
    execution, task, rows, original = owner_fixture(rounds=8, stream=stream)
    addresses = execution.arena.pool_addresses
    final = execution.step(rows, stream)
    context = task.context
    assert len(context.page_publications) == 16
    assert not context.page_writes and all(context.page_publications.values())
    assert all(release is not None for release in context.page_pins.values())
    if stream is not None:
        assert all(receipt.completion is not None for receipt in context.page_publications)
        assert all(
            pin.stream is stream and release.completion is not None for pin, release in context.page_pins.items()
        )
        torch.cuda.current_stream().wait_stream(stream)
    assert tuple(ref.version_id for ref in final) == (8, 8)
    for value, expected in zip(task.result.values, (1.0, 9.0, 10.0), strict=True):
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    if stream is None:  # Only the selected field was privately overwritten; others kept their pages.
        pin = execution.arena.pin_exact(final)
        for ref in final:
            for field in execution.arena.snapshot(pin, ref)[1:]:
                assert torch.count_nonzero(field) == 0
        execution.arena.wait_release(execution.arena.release_pin(pin))
    assert execution.arena.pool_addresses == addresses
    retire_all(execution, rows)
    # Returned old data survives retirement; released pages cannot be bound again.
    torch.testing.assert_close(task.result.values[0], torch.ones_like(task.result.values[0]), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="released"):
        task.region.run({}, pages=task.pages(context, original))


def test_page_bindings_reject_missing_cells_and_aliases_before_any_effect():
    execution, task, rows, refs = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    for cells, message in (
        ((ResourcePageCell("layers", (0,), refs[0], (LayoutFieldId(0),)),), "cover every"),
        (tuple(ResourcePageCell("layers", (i,), refs[0], (LayoutFieldId(0),)) for i in range(2)), "alias"),
    ):
        with pytest.raises(ValueError, match=message):
            RegionResources(task.program, invocation.context, cells)
    assert not invocation.context.page_publications
    with pytest.raises(RuntimeError, match="lease overlaps"):
        execution.start_step(rows)
    execution.cancel_step(invocation)
    execution.step(rows)
    retire_all(execution, rows)


def injected(*args):
    raise RuntimeError("injected failure")


def test_failures_keep_published_commits_and_abort_private_writes(monkeypatch):
    # A settlement failure after publication retains that receipt and cannot replay.
    execution, task, rows, _ = owner_fixture(rounds=2)
    invocation = execution.start_step(rows)
    anchor = execution.plan.tasks[0].anchor_id
    settle = execution.arena.settle
    monkeypatch.setattr(execution.arena, "settle", injected)
    with pytest.raises(StateSettlementError) as failed:
        execution.execute_anchor(invocation, anchor)
    receipt = failed.value.receipt
    assert receipt in invocation.context.page_publications
    assert execution.arena.latest(receipt.after_refs[0].lineage_id) == receipt.after_refs[0]
    with pytest.raises(StateSettlementError):
        execution.execute_anchor(invocation, anchor)
    assert len(invocation.context.page_publications) == 1
    monkeypatch.setattr(execution.arena, "settle", settle)
    execution.cancel_step(invocation)
    assert invocation.context.page_publications[receipt]
    assert execution.arena.usage().private_versions == 0
    execution.step(rows)  # A new invocation acquires the same directory lease.
    retire_all(execution, rows)
    # A write failure before publication aborts the private version.
    execution, task, rows, _ = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    write_field = execution.arena.write_field
    monkeypatch.setattr(execution.arena, "write_field", injected)
    with pytest.raises(RuntimeError, match="injected"):
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert not invocation.context.page_writes and invocation.context.release is not None
    assert execution.arena.usage().private_versions == 0
    monkeypatch.setattr(execution.arena, "write_field", write_field)
    execution.step(rows)
    retire_all(execution, rows)


def test_foreign_publication_cannot_silently_replace_a_snapshot_version():
    foreign, refs = [], []

    def wrap(arena, function):
        def changed(*args):
            if not foreign:
                # Bypass the request lease to test the arena's own expected-version check.
                write = arena.prepare_transition((refs[0],), fields=((LayoutFieldId(0),),), copy_fields=())
                arena.write_field(write, 0, LayoutFieldId(0)).fill_(99)
                receipt = arena.publish(write)
                arena.settle(receipt)
                foreign.append(receipt)
            return function(*args)

        return changed

    execution, task, rows, originals = owner_fixture(rounds=1, wrap=wrap)
    refs.extend(originals)
    # The serial schedule makes the foreign write land right after the first snapshot.
    task.region = bind(task.program, task.region.cells, analyze_region_program(task.program).reference_schedule)
    invocation = execution.start_step(rows)
    with pytest.raises(RuntimeError, match="not latest"):
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert not invocation.context.page_publications and invocation.context.release is not None
    pin = execution.arena.pin_exact(foreign[0].after_refs)
    value = execution.arena.snapshot(pin, pin.refs[0])[0]
    torch.testing.assert_close(value, torch.full_like(value, 99), rtol=0, atol=0)
    execution.arena.wait_release(execution.arena.release_pin(pin))


def test_returned_views_survive_page_reclamation_and_reuse():
    execution, task, rows, _ = owner_fixture(rounds=3, function=lambda w: (w.view_as(w),))
    execution.step(rows)
    retire_all(execution, rows)
    admitted = execution.arena.prepare_admission(("dense_fast_weights", "dense_fast_weights"))
    for root in execution.arena.writable_roots(admitted):
        root[0].fill_(99)
    for value, expected in zip(task.result.values, (1.0, 1.0, 2.0), strict=True):
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    execution.arena.abort(admitted)


def scan_snapshots(cell, count):
    """A scan inside a branch collecting the old and new version of layer 0."""
    tensor = cell.declaration.inputs[0]
    args = (Value("marker", I),)
    body = region(
        (Value("i", X), *args),
        (
            Coordinate((), (Value("zero", X),), AffineIndex(0)),
            Snapshot(("zero",), (Value("old", tensor),), "layers"),
            Call(("old",), (Value("new", tensor),), cell.digest),
            Publish(("new", "zero"), (), "layers"),
        ),
        ("marker", "old", "new"),
    )
    outputs = (Value("end", I), Value("olds", BankType(tensor, count)), Value("news", BankType(tensor, count)))
    branch = region(args, (Scan(("marker",), outputs, count, body),), tuple(v.name for v in outputs))
    entry = region(
        (*args, Value("p", B)), (If(("p", "marker"), outputs, branch, branch),), tuple(v.name for v in outputs)
    )
    return Program(entry, (cell.declaration,), (Resource("layers", (tensor,), (2,)),)), {"marker": 7, "p": True}


def while_publications(cell, limit, enabled):
    """A branch-guarded while loop publishing layer 0 once per iteration."""
    tensor = cell.declaration.inputs[0]
    args = (Value("n", I), Value("stop", I))
    condition = region(args, (Scalar(("n", "stop"), (Value("again", B),), ScalarKind.LESS),), ("again",))
    body = region(
        args,
        (
            Constant((), (Value("zero", X),), 0),
            Constant((), (Value("one", I),), 1),
            Snapshot(("zero",), (Value("w", tensor),), "layers"),
            Call(("w",), (Value("next_w", tensor),), cell.digest),
            Publish(("next_w", "zero"), (), "layers"),
            Scalar(("n", "one"), (Value("next_n", I),), ScalarKind.ADD),
        ),
        ("next_n", "stop"),
    )
    ends = (Value("end_n", I), Value("end_stop", I))
    branch = region(args, (While(("n", "stop"), ends, limit, condition, body),), ("end_n", "end_stop"))
    identity = region(args, (), ("n", "stop"))
    program = Program(
        region(
            (*args, Value("enabled", B)),
            (
                Constant((), (Value("zero", X),), 0),
                Snapshot(("zero",), (Value("old", tensor),), "layers"),
                If(("enabled", "n", "stop"), ends, branch, identity),
                Snapshot(("zero",), (Value("final", tensor),), "layers"),
            ),
            ("end_n", "old", "final"),
        ),
        (cell.declaration,),
        (Resource("layers", (tensor,), (2,)),),
    )
    return program, {"n": 0, "stop": 4, "enabled": enabled}


def test_control_flow_effects_match_the_reference_through_the_owner():
    for build, publications in (
        (lambda cell: scan_snapshots(cell, 3), 3),
        (lambda cell: while_publications(cell, 2, True), 2),  # Hits the iteration limit.
        (lambda cell: while_publications(cell, 8, True), 4),  # Stops on its predicate.
        (lambda cell: while_publications(cell, 8, False), 0),  # Branch not taken.
    ):
        execution, task, rows, _ = owner_fixture(rounds=1)
        (cell,) = task.region.cells.values()
        task.program, task.inputs = build(cell)
        task.region = bind(task.program, {cell.digest: cell})
        expected = interpret(
            task.program,
            task.inputs,
            resources=initial_layers(cell.input_signatures[0].shape),
            functions={cell.digest: lambda args, cell=cell: cell(*args)},
        )
        final = execution.step(rows)
        for got, want in zip(task.result.values, expected.values, strict=True):
            assert_same(got, want)
        assert tuple(ref.version_id for ref in final) == (publications, 0)
        assert len(task.context.page_publications) == publications
        retire_all(execution, rows)


def test_cancellation_inside_a_loop_keeps_the_first_commit_and_stops():
    execution, task, rows, _ = owner_fixture(rounds=1)
    (cell,) = task.region.cells.values()
    calls = []

    def cancel(*args):
        calls.append(1)
        if len(calls) == 2:
            task.context.cancel_requested.set()
        return cell.function(*args)

    tracked = replace(cell, function=cancel)
    task.program, task.inputs = while_publications(tracked, 8, True)
    task.inputs["stop"] = 6
    task.region = bind(task.program, {tracked.digest: tracked})
    invocation = execution.start_step(rows)
    with pytest.raises(StateSettlementError):
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert len(calls) == 2 and len(invocation.context.page_publications) == 1
    assert not invocation.context.page_writes
    execution.cancel_step(invocation)
    assert all(release is not None for release in invocation.context.page_pins.values())
    retire_all(execution, rows)


def native_scan_program(tensor, count):
    """A native import snapshots and publishes layer 0 inside a branch-guarded scan."""
    function = Function("native.test.page_update", (X,), (tensor,), ("layers",))
    body = region(
        (Value("i", X),),
        (Coordinate((), (Value("zero", X),), AffineIndex(0)), Call(("zero",), (Value("old", tensor),), function.name)),
        ("old",),
    )
    outputs = (Value("old_values", BankType(tensor, count)),)
    branch = region((), (Scan((), outputs, count, body),), ("old_values",))
    entry = region((Value("enabled", B),), (If(("enabled",), outputs, branch, branch),), ("old_values",))
    return Program(entry, (function,), (Resource("layers", (tensor,), (2,)),))


@pytest.mark.parametrize("cuda", (False, True))
def test_native_effects_read_exact_versions_on_the_owner_stream(cuda):
    if cuda and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    stream = torch.cuda.Stream() if cuda else None
    execution, task, rows, _ = owner_fixture(rounds=1, stream=stream)
    (exported,) = task.region.cells.values()
    t = exported.declaration.inputs[0]
    program = native_scan_program(t, 4)
    (declaration,) = program.functions
    views = []

    def update(inputs, resources):
        if stream is not None:
            assert torch.cuda.current_stream() == stream
        (index,) = inputs
        (old,) = resources.snapshot("layers", (index,))
        resources.publish("layers", (index,), (old + 2,))
        views.append(resources)
        return (old,)

    def oracle(inputs):
        index, family = inputs
        old = family.elements[index]
        return old, Bank((old + 2, family.elements[1]))

    task.program, task.inputs = program, {"enabled": True}
    task.region = bind(program, {declaration.name: NativePrimitive(declaration, update)})
    expected = interpret(
        program, task.inputs, functions={declaration.name: oracle}, resources=initial_layers(t.shape, t.device)
    )
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
    retire_all(execution, rows)


def test_native_failures_expire_their_scope_and_keep_the_commit_boundary():
    for fault in ("undeclared", "boolean_index", "bad_result", "cancel", "raise"):
        execution, task, rows, refs = owner_fixture(rounds=1)
        (exported,) = task.region.cells.values()
        program = native_scan_program(exported.declaration.inputs[0], 3)
        (declaration,) = program.functions
        views = []

        def fail(inputs, resources, fault=fault, task=task, views=views):
            views.append(resources)
            if fault == "undeclared":
                resources.snapshot("other", (0,))
            if fault == "boolean_index":
                resources.snapshot("layers", (False,))
            (old,) = resources.snapshot("layers", (0,))
            resources.publish("layers", (0,), (old + 1,))
            if fault == "bad_result":
                return (7,)
            if fault == "cancel":
                task.context.cancel_requested.set()
                return (old,)
            raise RuntimeError("native failure after commit")

        task.program, task.inputs = program, {"enabled": True}
        task.region = bind(program, {declaration.name: NativePrimitive(declaration, fail)})
        invocation = execution.start_step(rows)
        committed = fault not in ("undeclared", "boolean_index")
        with pytest.raises(StateSettlementError if committed else ValueError):
            execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
        assert len(views) == 1
        with pytest.raises(RuntimeError, match="expired"):
            views[0].snapshot("layers", (0,))
        assert execution.arena.latest(refs[0].lineage_id).version_id == int(committed), fault
        if committed:
            execution.cancel_step(invocation)
        else:
            with pytest.raises(RuntimeError):
                execution.cancel_step(invocation)
        assert not invocation.context.page_writes
        assert all(release is not None for release in invocation.context.page_pins.values())
        retire_all(execution, rows)


def payload_program(cell, rounds):
    """One atomic resource whose payload has four heterogeneous fields."""
    types = cell.declaration.inputs

    def values(prefix):
        return tuple(Value(f"{prefix}{i}", type_) for i, type_ in enumerate(types))

    old, new, initial, final = (values(prefix) for prefix in ("old", "new", "initial", "final"))
    body = region(
        (Value("i", X),),
        (
            Coordinate((), (Value("zero", X),), AffineIndex(0)),
            Snapshot(("zero",), old, "layers"),
            Call(tuple(v.name for v in old), new, cell.digest),
            Publish((*tuple(v.name for v in new), "zero"), (), "layers"),
        ),
        (),
    )
    return Program(
        region(
            (),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Snapshot(("zero",), initial, "layers"),
                Scan((), (), rounds, body),
                Snapshot(("zero",), final, "layers"),
            ),
            tuple(v.name for v in (*initial, *final)),
        ),
        (cell.declaration,),
        (Resource("layers", types, (2,)),),
    )


def configure_payload(execution, task, rounds):
    specs = execution.arena.plan.tensor_layout_request.layouts[0].layout.fields
    device = execution.arena.device
    examples = tuple(
        torch.zeros(s.shape, dtype=torch.int64 if i == 3 else torch.float32, device=device) for i, s in enumerate(specs)
    )

    def update(w, column, row, step):
        column = column + 1
        row = row + 2
        return w + column + row.T, column, row, step + 1

    cell = ExportedCell.capture(
        update, ("w", "column", "row", "step"), ("next_w", "next_column", "next_row", "next_step"), examples
    )
    task.program = payload_program(cell, rounds)
    task.region = bind(task.program, {cell.digest: cell})
    task.fields = tuple(LayoutFieldId(i) for i in range(len(specs)))
    return cell


@pytest.mark.parametrize("cuda", (False, True))
def test_heterogeneous_payload_fields_publish_atomically(cuda, monkeypatch):
    if cuda and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    stream = torch.cuda.Stream() if cuda else None
    execution, task, rows, refs = owner_fixture(rounds=1, stream=stream)
    configure_payload(execution, task, 5)
    final = execution.step(rows, stream)
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    # Independent closed form: step k adds 3k to every weight element.
    initial, actual = task.result.values[:4], task.result.values[4:]
    for tensor, number in zip(initial, (1, 0, 0, 0), strict=True):
        torch.testing.assert_close(tensor, torch.full_like(tensor, number), rtol=0, atol=0)
    for tensor, number in zip(actual, (46, 5, 10, 5), strict=True):
        torch.testing.assert_close(tensor, torch.full_like(tensor, number), rtol=0, atol=0)
    assert actual[3].dtype is torch.int64 and actual[3].shape == ()
    assert tuple(ref.version_id for ref in final) == (5, 0)
    assert len(task.context.page_publications) == 5  # One commit per payload, not per field.
    assert all(len(receipt.after_refs) == 1 for receipt in task.context.page_publications)
    retire_all(execution, rows)
    # A failure while copying the third field publishes none of them.
    execution, task, rows, refs = owner_fixture(rounds=1, stream=stream)
    configure_payload(execution, task, 3)
    write_field = execution.arena.write_field

    def fail(token, root, field):
        if field == 2:
            raise RuntimeError("injected third-field failure")
        return write_field(token, root, field)

    monkeypatch.setattr(execution.arena, "write_field", fail)
    invocation = execution.start_step(rows, stream)
    with pytest.raises(RuntimeError, match="third-field"):
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert not invocation.context.page_publications and not invocation.context.page_writes
    assert tuple(execution.arena.latest(ref.lineage_id) for ref in refs) == refs
    execution.retire(tuple(row.identity for row in rows))
    assert execution.arena.active_lineages == 0


CELL = AccessRange(AffineIndex(0, (IndexTerm(0),)), AffineIndex(1, (IndexTerm(0),)))
PREFIX = AccessRange(AffineIndex(), AffineIndex(0, (IndexTerm(0),)))


def interval_functions(tensor):
    read = Function("read", (X,), (tensor,), ("layers",), (ResourceAccess("layers", AccessMode.READ, (CELL,)),))
    write = Function("write", (X,), (), ("layers",), (ResourceAccess("layers", AccessMode.WRITE, (CELL,)),))
    prefix = Function("prefix", (X,), (tensor,), ("layers",), (ResourceAccess("layers", AccessMode.READ, (PREFIX,)),))
    return read, write, prefix


def test_declared_intervals_let_disjoint_writes_move_and_bound_native_access():
    execution, task, rows, _ = owner_fixture(rounds=1)
    (exported,) = task.region.cells.values()
    tensor = exported.declaration.inputs[0]
    read, write, prefix = interval_functions(tensor)
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
    callbacks = {
        "read": lambda inputs, resources: resources.snapshot("layers", (inputs[0],)),
        "write": lambda inputs, resources: (
            resources.publish("layers", (inputs[0],), (torch.full(tensor.shape, 10.0 + inputs[0]),)) or ()
        ),
        "prefix": lambda inputs, resources: (sum(resources.snapshot("layers", (i,))[0] for i in range(inputs[0])),),
    }
    cells = {f.name: NativePrimitive(f, callbacks[f.name]) for f in program.functions}
    # The write to layer 1 moves before the read of layer 0; RAW/WAR/WAW keep order.
    task.program = program
    task.region = bind(program, cells, "{ S0[] -> [1]; S1[] -> [0]; S2[] -> [2]; S3[] -> [3]; S4[] -> [4] }")
    final = execution.step(rows)
    old, total = task.result.values
    torch.testing.assert_close(old, torch.ones(tensor.shape), rtol=0, atol=0)
    torch.testing.assert_close(total, torch.full(tensor.shape, 21.0), rtol=0, atol=0)
    assert tuple(ref.version_id for ref in final) == (2, 1)
    assert len(task.context.page_publications) == 3
    retire_all(execution, rows)
    # A native import cannot read outside its interval, write a read-only one,
    # or receive an index outside the resource domain.
    for fault in ("read_other", "write_readonly", "outside_domain"):
        execution, task, rows, _ = owner_fixture(rounds=1)
        called = []

        def bad(inputs, resources, fault=fault, called=called):
            called.append(True)
            if fault == "write_readonly":
                resources.publish("layers", (0,), (torch.zeros(tensor.shape),))
            return resources.snapshot("layers", (1,))

        task.program = Program(
            region((Value("index", X),), (Call(("index",), (Value("old", tensor),), read.name),), ("old",)),
            (read,),
            (Resource("layers", (tensor,), (2,)),),
        )
        task.inputs = {"index": 2 if fault == "outside_domain" else 0}
        task.region = bind(task.program, {read.name: NativePrimitive(read, bad)})
        with pytest.raises(ValueError, match="declared"):
            execution.step(rows)
        assert bool(called) is (fault != "outside_domain")
        assert not task.context.page_publications
        retire_all(execution, rows)


@pytest.mark.parametrize("enabled", (False, True))
def test_conditional_nested_writes_only_fence_the_cells_they_touch(enabled):
    execution, task, rows, _ = owner_fixture(rounds=1)
    (exported,) = task.region.cells.values()
    tensor = exported.declaration.inputs[0]
    read, write, _ = interval_functions(tensor)
    leaf_args = (Value("index", X),)
    inner = region(
        (Value("j", X),),
        (
            Constant((), (Value("yes", B),), True),
            If(("yes", "j"), (), region(leaf_args, (Call(("index",), (), write.name),), ()), region(leaf_args, (), ())),
        ),
        (),
    )
    branch = region((), (Scan((), (), 2, region((Value("i", X),), (Scan((), (), 1, inner),), ())),), ())
    program = Program(
        region(
            (Value("enabled", B),),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Coordinate((), (Value("one", X),), AffineIndex(1)),
                If(("enabled",), (), branch, region((), (), ())),
                Call(("one",), (Value("other", tensor),), read.name),
                Call(("zero",), (Value("changed", tensor),), read.name),
            ),
            ("other", "changed"),
        ),
        (read, write),
        (Resource("layers", (tensor,), (2,)),),
    )
    with pytest.raises(ValueError, match="dependency"):
        bind(program, schedule="{ S0[] -> [1]; S1[] -> [2]; S2[] -> [0] }")
    calls = []

    def read_(inputs, resources):
        calls.append(("read", inputs[0]))
        return resources.snapshot("layers", (inputs[0],))

    def write_(inputs, resources):
        calls.append(("write", inputs[0]))
        resources.publish("layers", (inputs[0],), (torch.full(tensor.shape, 10.0),))
        return ()

    task.program, task.inputs = program, {"enabled": enabled}
    cells = {read.name: NativePrimitive(read, read_), write.name: NativePrimitive(write, write_)}
    task.region = bind(program, cells, "{ S0[] -> [1]; S1[] -> [0]; S2[] -> [2] }")
    final = execution.step(rows)
    assert calls == [("read", 1), *([("write", 0)] * 2 if enabled else []), ("read", 0)]
    for actual, expected in zip(task.result.values, (2.0, 10.0 if enabled else 1.0), strict=True):
        torch.testing.assert_close(actual, torch.full(tensor.shape, expected), rtol=0, atol=0)
    assert tuple(ref.version_id for ref in final) == (2 if enabled else 0, 0)
    retire_all(execution, rows)


def test_parametric_publications_use_the_size_binding_of_their_invocation(monkeypatch):
    execution, task, rows, _ = owner_fixture(rounds=1)
    (cell,) = task.region.cells.values()
    program = replace(page_program(cell, rounds="rounds"), parameters=(SizeParameter("rounds", 0, 5),))
    runner = bind(program, {cell.digest: cell})
    active = {"rounds": 0}
    contexts, results = [], []

    def pages(context, sizes):
        cells = tuple(
            ResourcePageCell("layers", (i,), ref, (LayoutFieldId(0),)) for i, ref in enumerate(context.pin.refs)
        )
        return RegionResources(program, context, cells, sizes=sizes)

    def run(context):
        contexts.append(context)
        results.append(runner.run({}, pages=pages(context, active), sizes=active))

    monkeypatch.setattr(task, "execute", run)
    total = 0
    for count in (0, 5, 2):
        active["rounds"] = count
        final = execution.step(rows)
        assert len(contexts[-1].page_publications) == count * 2
        total += count
        assert tuple(ref.version_id for ref in final) == (total, total)
        for result, expected in zip(results[-1].values[1:], (total + 1.0, total + 2.0), strict=True):
            torch.testing.assert_close(result, torch.full_like(result, expected), rtol=0, atol=0)
    invocation = execution.begin(rows)
    with pytest.raises(ValueError, match="different size"):
        runner.run({}, pages=pages(invocation.context, {"rounds": 1}), sizes={"rounds": 2})
    assert not invocation.context.page_publications and not invocation.context.page_writes
    execution.cancel_step(invocation)
    retire_all(execution, rows)


# ---------------------------------------------------------------------------
# Several logical resources in one arena, admission and invocation owner
# ---------------------------------------------------------------------------

DENSE = StateResource("dense_fast_weights")
AUX = StateResource("fast_statistics")
GLOBAL = StateResource("global_factors")


def multistate_arena(*, route=False, instance=True):
    """Request dense weights, request/route statistics and optional instance factors."""
    plan = page_arena(heterogeneous=True).plan
    dense, global_ = plan.tensor_layout_request.layouts
    aux_layout = replace(
        global_.layout, namespace="test.auxiliary@1", fields=(replace(global_.layout.fields[0], shape=(3,)),)
    )
    layouts = TensorLayoutRequest((dense, StateTensorLayout(str(AUX), aux_layout), *((global_,) if instance else ())))
    states = plan.semantic_graph.states
    aux = SemanticState(AUX, StateScope.REQUEST, (RequestKeyField.ROUTE,) if route else states[0].key_fields)
    graph = replace(plan.semantic_graph, states=(states[0], aux, *((states[1],) if instance else ())))
    return PageBundleArena(replan_states(plan, graph, layouts, 8192))


@dataclass
class ResourceTask:
    implementation_id: int
    runner: RegionExecution | None = None
    program: Program | None = None
    context = None

    def execute(self, context):
        if self.runner is None:
            return
        self.context = context
        cells = tuple(
            ResourcePageCell(str(group.resource), (i,), ref, (LayoutFieldId(0),))
            for group in context.selection.groups
            for i, ref in enumerate(group.refs)
        )
        self.runner.run({}, pages=RegionResources(self.program, context, cells))


def multistate(*, route=False, instance=True):
    arena = multistate_arena(route=route, instance=instance)
    initialized = []

    def initializer(name, value):
        def initialize(roots):
            initialized.append((name, len(roots)))
            for root in roots:
                for field in root:
                    field.fill_(value)

        return initialize

    functions = FunctionMath(
        {
            str(state.resource): initializer(state.resource, value)
            for state, value in zip(arena.plan.semantic_graph.states, (1, 10, 100))
        },
        lambda rows: rows,
        lambda rows, batch: rows,
        lambda rows, output: output.selection,
    )
    tasks = tuple(ResourceTask(task.implementation_id) for task in arena.plan.schedule_physical.tasks)
    execution = MathBinding(functions, tasks).bind(arena, AnchorId(arena.plan.timeline.anchors[-1].ordinal))
    return execution, tasks[0], initialized


def rows_for(*identities):
    return tuple(PackedPhaseRow(i, i, 0, HiddenPhase.PREFILL, identity) for i, identity in enumerate(identities))


def increment_every_resource(task, execution, rows):
    """One region scans each selected resource family and adds one to every cell."""
    resources, operations, functions = [], [], {}
    for group in execution.directory.select(rows).groups:
        name = str(group.resource)
        cell = ExportedCell.capture(lambda w: (w + 1,), ("w",), ("new",), (execution.snapshot(group.refs[0])[0],))
        functions[cell.digest] = cell
        tensor = cell.declaration.inputs[0]
        body = region(
            (Value("i", X),),
            (
                Snapshot(("i",), (Value("old", tensor),), name),
                Call(("old",), (Value("new", tensor),), cell.digest),
                Publish(("new", "i"), (), name),
            ),
            (),
        )
        resources.append(Resource(name, (tensor,), (len(group.refs),)))
        operations.append(Scan((), (), len(group.refs), body))
    task.program = Program(
        region((), tuple(operations), ()), tuple(cell.declaration for cell in functions.values()), tuple(resources)
    )
    task.runner = bind(task.program, functions, analyze_region_program(task.program).reference_schedule)


def assert_multistate_reclaimed(execution):
    execution.arena.collect_completed(wait=True)
    usage = execution.arena.usage()
    assert execution.directory.active_count == execution.arena.active_lineages == 0
    assert execution.directory.inference_pin_count == 0
    assert usage.private_versions == usage.pinned_versions == usage.pending_receipts == 0


def test_heterogeneous_resources_admit_publish_and_pin_in_one_invocation(monkeypatch):
    execution, task, initialized = multistate()
    (global_ref,) = execution.admit(())
    a, b = RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0)
    publications = []
    publish = execution.arena.publish
    monkeypatch.setattr(
        execution.arena, "publish", lambda write: publications.append(publish(write)) or publications[-1]
    )
    refs = execution.admit((a, b))
    assert len(publications) == 1 and len(publications[0].after_refs) == 4  # One atomic admission.
    assert initialized == [(GLOBAL, 1), (DENSE, 2), (AUX, 2)]
    rows = rows_for(a, b)
    selected = execution.directory.select(rows)
    assert selected.for_resource(DENSE).keys == selected.for_resource(AUX).keys
    assert selected.ref_for(b, AUX) == refs[3] and selected.ref_for(a, GLOBAL) == global_ref
    with pytest.raises(ValueError, match="name a resource"):
        selected.ref_for(a)
    increment_every_resource(task, execution, rows)
    execution.step(rows)
    context = task.context
    assert len(context.page_publications) == 5 and not context.page_writes
    for old, expected in zip(refs, (2, 2, 11, 11, 101), strict=True):
        current = context.current_owned_ref(old)
        assert current.version_id == old.version_id + 1
        value = execution.snapshot(current)[0]
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    # Request inference still owns the original global version after publication.
    pin = execution.directory.inference_pin(a)
    assert pin.refs == (global_ref,)
    torch.testing.assert_close(execution.arena.snapshot(pin, global_ref)[0], torch.full((2, 5), 100.0), rtol=0, atol=0)
    execution.retire((a, b))
    execution.retire(())
    assert_multistate_reclaimed(execution)


def test_admission_failure_rolls_back_every_resource_and_inference_pin(monkeypatch):
    for failure in ("initialize", "pin", "publish"):
        execution, _, _ = multistate()
        execution.admit(())
        a, b = RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0)

        def fail(*args):
            raise RuntimeError("injected admission failure")

        with monkeypatch.context() as changed:
            if failure == "initialize":
                changed.setitem(execution.math.initializers, str(AUX), fail)
            elif failure == "pin":
                pin_exact, count = execution.arena.pin_exact, []
                changed.setattr(
                    execution.arena,
                    "pin_exact",
                    lambda *args: fail() if count.append(1) or len(count) == 2 else pin_exact(*args),
                )
            else:
                changed.setattr(execution.arena, "publish", fail)
            with pytest.raises(RuntimeError, match="injected admission"):
                execution.admit((a, b))
        execution.arena.collect_completed(wait=True)
        assert execution.directory.active_count == execution.arena.active_lineages == 1, failure
        assert execution.directory.inference_pin_count == 0
        assert execution.arena.usage().private_versions == execution.arena.usage().pinned_versions == 0
        execution.admit((a, b))
        execution.retire((a, b))
        execution.retire(())
        assert_multistate_reclaimed(execution)


def test_shared_route_state_leases_exclusively_and_persists_until_dormant_retirement():
    execution, _, _ = multistate(route=True, instance=False)
    a, b = RequestIdentity("a", 0, 0, "math"), RequestIdentity("b", 1, 0, "math")
    execution.admit((a,))
    joined = execution.admit((b,))
    left, right = execution.directory.select(rows_for(a)), execution.directory.select(rows_for(b))
    assert left.ref_for(a, AUX) == right.ref_for(b, AUX) == joined[-1]
    invocation = execution.begin(rows_for(a))
    with pytest.raises(RuntimeError, match="lease overlaps"):
        execution.begin(rows_for(b))
    execution.cancel_step(invocation)
    invocation = execution.begin(rows_for(b))  # No partial lease leaked for b's own dense resource.
    execution.retire((a,))
    assert invocation.context.release is None
    execution.cancel_step(invocation)
    execution.retire((b,))
    assert execution.directory.active_count == 1  # The shared route persists dormant.
    c = RequestIdentity("c", 2, 0, "math")
    assert execution.admit((c,))[-1] == joined[-1]
    execution.retire((c,))
    execution.retire_dormant()
    assert_multistate_reclaimed(execution)


def test_failure_between_resource_publications_keeps_the_committed_prefix(monkeypatch):
    execution, task, _ = multistate()
    execution.admit(())
    a = RequestIdentity("a", 0, 0)
    refs = execution.admit((a,))
    rows = rows_for(a)
    increment_every_resource(task, execution, rows)
    invocation = execution.start_step(rows)
    write_field = execution.arena.write_field

    def fail_second(token, index, field):
        if token.before_refs[0].lineage_id == refs[1].lineage_id:
            raise RuntimeError("injected second resource failure")
        return write_field(token, index, field)

    with monkeypatch.context() as changed:
        changed.setattr(execution.arena, "write_field", fail_second)
        for _ in range(2):  # Retrying the anchor cannot republish the first resource.
            with pytest.raises(StateSettlementError):
                execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
            assert len(invocation.context.page_publications) == 1
    assert tuple(execution.arena.latest(ref.lineage_id).version_id for ref in refs) == (1, 0, 0)
    execution.cancel_step(invocation)
    assert not invocation.context.page_writes
    execution.step(rows)
    execution.retire((a,))
    execution.retire(())
    assert_multistate_reclaimed(execution)


def test_retiring_a_request_cancels_its_invocation_and_releases_only_its_pins():
    execution, _, _ = multistate(route=True)
    (global_ref,) = execution.admit(())
    a, b = RequestIdentity("a", 0, 0, "math"), RequestIdentity("b", 1, 0, "math")
    execution.admit((a, b))
    invocation = execution.begin(rows_for(a))
    with pytest.raises(RuntimeError, match="inference requests"):
        execution.retire(())
    assert invocation.context.release is None
    execution.retire((a,))
    assert invocation.context.cancel_requested.is_set() and invocation.context.release is not None
    assert not execution.directory.has_inference_pin(a)
    assert execution.directory.inference_pin(b).refs == (global_ref,)
    execution.cancel_step(execution.begin(rows_for(b)))
    execution.retire((b,))
    # An offline instance invocation has no request roots, even in a mixed directory.
    invocation = execution.begin(())
    assert invocation.context.pin.refs == (global_ref,)
    assert invocation.context.selection.for_resource(DENSE).refs == ()
    execution.retire(())
    assert invocation.context.release is not None
    execution.retire_dormant()
    assert_multistate_reclaimed(execution)
