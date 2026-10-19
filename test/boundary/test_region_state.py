"""Region effects on PTTS state: the request owner's leases, COW pages and commits.

Programs snapshot and publish request pages through ``RegionResources``. Every
scenario runs through ``StateExecution`` and checks the user-visible PTTS
contract: published versions, pins, failure/cancel commit
boundaries and full reclamation after retirement. The multistate scenarios
share one arena between request, route and instance resources.
"""

from dataclasses import dataclass, replace

import pytest
import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_program import (
    AffineIndex,
    BankType,
    Call,
    Constant,
    Coordinate,
    Function,
    If,
    Program,
    Publish,
    Resource,
    Scalar,
    ScalarKind,
    ScalarType,
    Scan,
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
    final = execution.step(rows, stream)
    if stream is not None:
        torch.cuda.current_stream().wait_stream(stream)
    assert tuple(ref.version_id for ref in final) == (8, 8)
    for value, expected in zip(task.result.values, (1.0, 9.0, 10.0), strict=True):
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    retire_all(execution, rows)
    # Returned old data survives retirement; released pages cannot be bound again.
    torch.testing.assert_close(task.result.values[0], torch.ones_like(task.result.values[0]), rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        task.region.run({}, pages=task.pages(task.context, original))


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
        retire_all(execution, rows)


def test_cancellation_inside_a_loop_keeps_the_first_commit_and_stops():
    execution, task, rows, refs = owner_fixture(rounds=1)
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
    assert len(calls) == 2 and execution.arena.latest(refs[0].lineage_id).version_id == 1
    execution.cancel_step(invocation)
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

    def update(inputs, resources):
        if stream is not None:
            assert torch.cuda.current_stream() == stream
        (index,) = inputs
        (old,) = resources.snapshot("layers", (index,))
        resources.publish("layers", (index,), (old + 2,))
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
    retire_all(execution, rows)


def test_native_failures_expire_their_scope_and_keep_the_commit_boundary():
    for fault in ("undeclared", "raise"):
        execution, task, rows, refs = owner_fixture(rounds=1)
        (exported,) = task.region.cells.values()
        program = native_scan_program(exported.declaration.inputs[0], 3)
        (declaration,) = program.functions
        views = []

        def fail(inputs, resources, fault=fault, task=task, views=views):
            views.append(resources)
            if fault == "undeclared":
                resources.snapshot("other", (0,))
            (old,) = resources.snapshot("layers", (0,))
            resources.publish("layers", (0,), (old + 1,))
            raise RuntimeError("native failure after commit")

        task.program, task.inputs = program, {"enabled": True}
        task.region = bind(program, {declaration.name: NativePrimitive(declaration, fail)})
        invocation = execution.start_step(rows)
        committed = fault == "raise"
        with pytest.raises(StateSettlementError if committed else ValueError):
            execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
        assert len(views) == 1
        with pytest.raises(RuntimeError):  # The native scope expired with the call.
            views[0].snapshot("layers", (0,))
        assert execution.arena.latest(refs[0].lineage_id).version_id == int(committed), fault
        if committed:
            execution.cancel_step(invocation)
        else:
            with pytest.raises(RuntimeError):
                execution.cancel_step(invocation)
        retire_all(execution, rows)


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


def multistate(*, route=False, instance=True, failing=()):
    arena = multistate_arena(route=route, instance=instance)

    def initializer(name, value):
        def initialize(roots):
            if name in failing:
                raise RuntimeError("injected admission failure")
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
    return execution, tasks[0]


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


def test_heterogeneous_resources_admit_publish_and_pin_in_one_invocation():
    execution, task = multistate()
    (global_ref,) = execution.admit(())
    a, b = RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0)
    refs = execution.admit((a, b))
    rows = rows_for(a, b)
    selected = execution.directory.select(rows)
    assert selected.ref_for(b, AUX) == refs[3] and selected.ref_for(a, GLOBAL) == global_ref
    increment_every_resource(task, execution, rows)
    execution.step(rows)
    context = task.context
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


def test_admission_failure_rolls_back_every_resource_and_inference_pin():
    execution, _ = multistate(failing=(AUX,))
    execution.admit(())
    a, b = RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0)
    with pytest.raises(RuntimeError):
        execution.admit((a, b))
    execution.arena.collect_completed(wait=True)
    assert execution.directory.active_count == execution.arena.active_lineages == 1
    assert execution.directory.inference_pin_count == 0
    assert execution.arena.usage().private_versions == execution.arena.usage().pinned_versions == 0
    execution.retire(())
    assert_multistate_reclaimed(execution)


def test_shared_route_state_leases_exclusively_and_persists_until_dormant_retirement():
    execution, _ = multistate(route=True, instance=False)
    a, b = RequestIdentity("a", 0, 0, "math"), RequestIdentity("b", 1, 0, "math")
    execution.admit((a,))
    joined = execution.admit((b,))
    left, right = execution.directory.select(rows_for(a)), execution.directory.select(rows_for(b))
    assert left.ref_for(a, AUX) == right.ref_for(b, AUX) == joined[-1]
    invocation = execution.begin(rows_for(a))
    with pytest.raises(RuntimeError):  # A single writer holds the shared route state.
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


def test_retiring_a_request_cancels_its_invocation_and_releases_only_its_pins():
    execution, _ = multistate(route=True)
    (global_ref,) = execution.admit(())
    a, b = RequestIdentity("a", 0, 0, "math"), RequestIdentity("b", 1, 0, "math")
    execution.admit((a, b))
    invocation = execution.begin(rows_for(a))
    with pytest.raises(RuntimeError):  # Instance state outlives the requests still pinning it.
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
