"""Multiple logical resources share one arena, admission and invocation owner."""

from dataclasses import dataclass, replace

import pytest
import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_program import Call, Program, Publish, Resource, ScalarType, Scan, Snapshot, Value
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.semantic_graph import AnchorId, SemanticState, StateResource, StateScope
from pllm.contracts.state_representation import StateLayoutRequirement, StateRepresentationRequest
from pllm.contracts.tensor_layout import LayoutFieldId, StateTensorLayout, TensorLayoutRequest
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_resources import RegionResources, ResourcePageCell
from pllm.execution.scheduler import compile_state_schedule, intern_tensor_layouts
from pllm.execution.state_execution import MathBinding, StateSettlementError
from pllm.execution.tensor_program import FunctionMath, _read
from test.boundary.test_page_bundle import _arena
from test.boundary.test_region_program import region

DENSE = StateResource("dense_fast_weights")
AUX = StateResource("fast_statistics")
GLOBAL = StateResource("global_factors")


def multistate_arena(*, route=False, instance=True):
    plan = _arena(heterogeneous=True).plan
    dense, global_ = plan.tensor_layout_request.layouts
    layouts = TensorLayoutRequest(
        (
            dense,
            StateTensorLayout(
                str(AUX),
                replace(
                    global_.layout,
                    namespace="test.auxiliary@1",
                    fields=(replace(global_.layout.fields[0], shape=(3,)),),
                ),
            ),
            *((global_,) if instance else ()),
        )
    )
    states = plan.semantic_graph.states
    graph = replace(
        plan.semantic_graph,
        states=(
            states[0],
            SemanticState(AUX, StateScope.REQUEST, (RequestKeyField.ROUTE,) if route else states[0].key_fields),
            *((states[1],) if instance else ()),
        ),
    )
    bindings = intern_tensor_layouts(layouts)
    provider = PageBundleProviderFacts("test.multistate@1", 6)
    candidates = tuple(
        provider.materialize_plan(
            resource=entry.resource, layout=entry.layout, binding=binding, max_admitted_lineages=2
        )
        for entry, binding in zip(layouts.layouts, bindings.bindings, strict=True)
    )
    request = StateRepresentationRequest(
        8192,
        tuple(
            StateLayoutRequirement(
                binding.resource,
                binding.layout_id,
                binding.layout_namespace,
                binding.layout_digest,
                2,
            )
            for binding in bindings.bindings
        ),
        candidates,
    )
    decision, physical, schedule, scheduled = compile_state_schedule(graph, plan.timeline, request)
    return PageBundleArena(
        replace(
            plan,
            semantic_graph=graph,
            schedule_decision=schedule,
            schedule_physical=scheduled,
            state_request=request,
            state_decision=decision,
            state_physical=physical,
            tensor_layout_request=layouts,
            tensor_layout_decision=bindings,
        )
    )


@dataclass
class Task:
    implementation_id: int
    runner: RegionExecution | None = None
    program: Program | None = None
    context = None
    result = None

    def execute(self, context):
        if self.runner is None:
            return
        self.context = context
        pages = RegionResources(
            self.program,
            context,
            tuple(
                ResourcePageCell(str(group.resource), (i,), ref, (LayoutFieldId(0),))
                for group in context.selection.groups
                for i, ref in enumerate(group.refs)
            ),
        )
        self.result = self.runner.run({}, pages=pages)


def fixture(*, route=False, instance=True):
    arena = multistate_arena(route=route, instance=instance)
    initialized, batches = [], []

    def initializer(name, value):
        def initialize(roots):
            initialized.append((name, len(roots)))
            for root in roots:
                for field in root:
                    field.fill_(value)

        return initialize

    def prepare(rows, batch):
        batches.append(batch)
        return rows

    functions = FunctionMath(
        {
            str(state.resource): initializer(state.resource, value)
            for state, value in zip(arena.plan.semantic_graph.states, (1, 10, 100))
        },
        lambda rows: rows,
        prepare,
        lambda rows, output: output.selection,
    )
    tasks = tuple(Task(task.implementation_id) for task in arena.plan.schedule_physical.tasks)
    execution = MathBinding(functions, tasks).bind(arena, AnchorId(arena.plan.timeline.anchors[-1].ordinal))
    return execution, tasks[0], initialized, batches


def rows_for(*identities):
    return tuple(PackedPhaseRow(i, i, 0, HiddenPhase.PREFILL, identity) for i, identity in enumerate(identities))


def bind_region(task, execution, rows):
    selection = execution.directory.select(rows)
    resources, operations, functions = [], [], {}
    for group in selection.groups:
        name = str(group.resource)
        example = execution.snapshot(group.refs[0])[0]
        cell = ExportedCell.capture(lambda w: (w + 1,), ("w",), ("new",), (example,))
        functions[cell.digest] = cell
        tensor = cell.declaration.inputs[0]

        body = region(
            (Value("i", ScalarType.INDEX),),
            (
                Snapshot(("i",), (Value("old", tensor),), name),
                Call(("old",), (Value("new", tensor),), cell.digest),
                Publish(("new", "i"), (), name),
            ),
            (),
        )
        resources.append(Resource(name, (tensor,), (len(group.refs),)))
        operations.append(Scan((), (), len(group.refs), body))
    program = Program(
        region((), tuple(operations), ()), tuple(cell.declaration for cell in functions.values()), tuple(resources)
    )
    schedule = analyze_region_program(program).reference_schedule
    task.runner = RegionExecution(
        program, lower_region_schedule(program, schedule), lower_region_bindings(program, schedule), functions
    )
    task.program = program


def assert_reclaimed(execution):
    execution.arena.collect_completed(wait=True)
    usage = execution.arena.usage()
    assert execution.directory.active_count == execution.arena.active_lineages == 0
    assert execution.directory.inference_pin_count == 0
    assert usage.private_versions == usage.pinned_versions == usage.pending_receipts == 0


def test_heterogeneous_resources_run_in_one_original_invocation(monkeypatch):
    execution, task, initialized, batches = fixture()
    (global_ref,) = execution.admit(())
    a, b = RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0)
    publications = []
    publish = execution.arena.publish

    def record(write):
        receipt = publish(write)
        publications.append(receipt)
        return receipt

    monkeypatch.setattr(execution.arena, "publish", record)
    refs = execution.admit((a, b))
    assert len(publications) == 1 and len(publications[0].after_refs) == 4
    assert initialized == [(GLOBAL, 1), (DENSE, 2), (AUX, 2)]
    rows = rows_for(a, b)
    selected = execution.directory.select(rows)
    assert selected.for_resource(DENSE).keys == selected.for_resource(AUX).keys
    assert len(set(selected.addresses)) == len(set(selected.refs)) == 5
    assert selected.for_resource(AUX).state_indices == (2, 3)
    assert selected.ref_for(b, AUX) == refs[3]
    assert selected.ref_for(a, GLOBAL) == global_ref
    with pytest.raises(ValueError, match="name a resource"):
        selected.ref_for(a)
    bind_region(task, execution, rows)
    execution.step(rows)
    assert tuple(len(root) for root in batches[0].roots) == (4, 4, 1, 1, 1)
    context = task.context
    assert len(context.page_publications) == 5
    assert all(context.page_publications.values()) and not context.page_writes
    for old, expected in zip(refs, (2, 2, 11, 11, 101), strict=True):
        current = context.current_owned_ref(old)
        assert current.version_id == old.version_id + 1
        torch.testing.assert_close(
            execution.snapshot(current)[0], torch.full_like(execution.snapshot(current)[0], expected), rtol=0, atol=0
        )
    # Request inference still owns the original global version after publication.
    pin = execution.directory.inference_pin(a)
    assert pin.refs == (global_ref,)
    torch.testing.assert_close(execution.arena.snapshot(pin, global_ref)[0], torch.full((2, 5), 100.0), rtol=0, atol=0)
    execution.retire((a, b))
    execution.retire(())
    assert_reclaimed(execution)


@pytest.mark.parametrize("failure", ("initialize", "pin", "publish"))
def test_admission_failure_rolls_back_all_resources_and_inference_pins(monkeypatch, failure):
    execution, _, _, _ = fixture()
    execution.admit(())
    a, b = RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0)

    def fail(*args):
        raise RuntimeError("injected admission failure")

    with monkeypatch.context() as changed:
        if failure == "initialize":
            changed.setitem(execution.math.initializers, str(AUX), fail)
        elif failure == "pin":
            pin_exact, count = execution.arena.pin_exact, 0

            def pin(*args):
                nonlocal count
                count += 1
                return fail() if count == 2 else pin_exact(*args)

            changed.setattr(execution.arena, "pin_exact", pin)
        else:
            changed.setattr(execution.arena, "publish", fail)
        with pytest.raises(RuntimeError, match="injected admission"):
            execution.admit((a, b))
    execution.arena.collect_completed(wait=True)
    assert execution.directory.active_count == execution.arena.active_lineages == 1
    assert execution.directory.inference_pin_count == 0
    assert execution.arena.usage().private_versions == execution.arena.usage().pinned_versions == 0
    execution.admit((a, b))
    execution.retire((a, b))
    execution.retire(())
    assert_reclaimed(execution)


def test_route_overlap_does_not_acquire_partial_lease_or_drop_other_holders():
    execution, _, _, _ = fixture(route=True, instance=False)
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
    assert execution.directory.active_count == 1  # Shared route persists dormant.
    c = RequestIdentity("c", 2, 0, "math")
    assert execution.admit((c,))[-1] == joined[-1]
    execution.retire((c,))
    execution.retire_dormant()
    assert_reclaimed(execution)


def test_failure_between_resource_publications_keeps_committed_prefix(monkeypatch):
    execution, task, _, _ = fixture()
    execution.admit(())
    a = RequestIdentity("a", 0, 0)
    refs = execution.admit((a,))
    rows = rows_for(a)
    bind_region(task, execution, rows)
    invocation = execution.start_step(rows)
    original = execution.arena.write_field

    def fail_second(token, index, field):
        if token.before_refs[0].lineage_id == refs[1].lineage_id:
            raise RuntimeError("injected second resource failure")
        return original(token, index, field)

    with monkeypatch.context() as changed:
        changed.setattr(execution.arena, "write_field", fail_second)
        with pytest.raises(StateSettlementError):
            execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
        assert len(invocation.context.page_publications) == 1
        with pytest.raises(StateSettlementError):
            execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
        assert len(invocation.context.page_publications) == 1
    assert tuple(execution.arena.latest(ref.lineage_id).version_id for ref in refs) == (1, 0, 0)
    with pytest.raises(ValueError, match="explicit Region"):
        _read(invocation.context, "published")
    execution.cancel_step(invocation)
    assert not invocation.context.page_writes
    execution.step(rows)
    execution.retire((a,))
    execution.retire(())
    assert_reclaimed(execution)


def test_mixed_retirement_cancels_own_invocation_then_releases_instance_pin():
    execution, _, _, _ = fixture(route=True)
    (global_ref,) = execution.admit(())
    a, b = RequestIdentity("a", 0, 0, "math"), RequestIdentity("b", 1, 0, "math")
    execution.admit((a, b))
    invocation = execution.begin(rows_for(a))
    with pytest.raises(RuntimeError, match="inference requests"):
        execution.retire(())
    assert invocation.context.release is None
    execution.retire((a,))
    assert invocation.context.cancel_requested.is_set()
    assert invocation.context.release is not None
    assert not execution.directory.has_inference_pin(a)
    assert execution.directory.inference_pin(b).refs == (global_ref,)
    invocation = execution.begin(rows_for(b))
    execution.cancel_step(invocation)
    execution.retire((b,))
    # An offline instance invocation has no request roots, even in a mixed directory.
    invocation = execution.begin(())
    assert invocation.context.pin.refs == (global_ref,)
    assert invocation.context.selection.for_resource(DENSE).refs == ()
    execution.retire(())
    assert invocation.context.release is not None
    execution.retire_dormant()
    assert_reclaimed(execution)
