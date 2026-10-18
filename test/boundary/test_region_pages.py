"""Region effects use the existing request lease, COW arena and failure owner."""

from dataclasses import dataclass, replace

import pytest
import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_program import (
    AffineIndex,
    Call,
    Coordinate,
    Program,
    Publish,
    Resource,
    ScalarType,
    Scan,
    Snapshot,
    Value,
)
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution
from pllm.execution.region_resources import RegionResources, ResourcePageCell
from pllm.execution.state_execution import StateExecution, StateSettlementError
from test.boundary.test_page_bundle import _arena
from test.boundary.test_region_program import region

X = ScalarType.INDEX


def page_program(cell, rounds=8):
    tensor = cell.declaration.inputs[0]

    layer = region(
        (Value("l", X),),
        (
            Snapshot(("l",), (Value("w", tensor),), "layers"),
            Call(("w",), (Value("new", tensor),), cell.digest),
            Publish(("new", "l"), (), "layers"),
        ),
        (),
    )
    chunk = region((Value("c", X),), (Scan((), (), 2, layer),), ())
    return Program(
        region(
            (),
            (
                Coordinate((), (Value("zero", X),), AffineIndex(0)),
                Coordinate((), (Value("one", X),), AffineIndex(1)),
                Snapshot(("zero",), (Value("old", tensor),), "layers"),
                Scan((), (), rounds, chunk),
                Snapshot(("zero",), (Value("first", tensor),), "layers"),
                Snapshot(("one",), (Value("second", tensor),), "layers"),
            ),
            ("old", "first", "second"),
        ),
        (cell.declaration,),
        (Resource("layers", (tensor,), (2,)),),
    )


class Math:
    def initialize(self, roots, resources):
        for i, root in enumerate(roots):
            for field in root:
                field.zero_()
            root[0].fill_(i + 1)

    def rows(self, value):
        return value

    def prepare(self, value, context):
        return value

    def output(self, context):
        return tuple(context.current_owned_ref(ref) for ref in context.pin.refs)


@dataclass
class Task:
    implementation_id: int
    region: RegionExecution | None = None
    program: Program | None = None
    result = None
    context = None
    inputs = None
    fields = (LayoutFieldId(0),)

    def execute(self, context):
        if self.region is not None:
            self.context = context
            pages = RegionResources(
                self.program,
                context,
                tuple(ResourcePageCell("layers", (i,), ref, self.fields) for i, ref in enumerate(context.pin.refs)),
            )
            self.result = self.region.run({} if self.inputs is None else self.inputs, pages=pages)


def owner_fixture(*, rounds=8, wrap=None, stream=None, function=None):
    arena = _arena()
    if stream is not None:
        from pllm.authoring import BackendCompileFacts, ParameterTargetShape
        from pllm.contracts.backend_attachment import (
            AttachmentGraphMode,
            AttachmentTopology,
            BackendPlacementProbe,
        )
        from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
        from pllm.contracts.state_representation import StateDeviceKind
        from pllm.contracts.tensor_layout import TensorDType
        from pllm.execution.ptts.page_bundle import PageBundleArena
        from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
        from pllm.execution.runtime_facts import RuntimeFacts
        from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload

        target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
        facts = BackendCompileFacts(
            BackendPlacementProbe(
                "test.region.cuda@1",
                "a" * 64,
                "b" * 64,
                "c" * 64,
                AttachmentTopology.IN_PROCESS,
                AttachmentGraphMode.EAGER,
                StateDeviceKind.CUDA,
                0,
                1,
                1,
                TensorDType.FLOAT32,
                3,
                8,
                (),
                (),
            ),
            RuntimeFacts(stream.device).timeline(("transition/full_root@1", "state/fenced_reclaim@1")),
            (PageBundleProviderFacts("test.region.cuda@1", 6, StateDeviceKind.CUDA),),
            2,
            4096,
            0,
            (ParameterTargetShape(target, 3, 2),),
        )
        workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=2))
        arena = PageBundleArena(workload.plan_backend(facts).inputs.plan)
    spec = arena.plan.tensor_layout_request.layouts[0].layout.fields[0]
    cell = ExportedCell.capture(
        (lambda w: (w + 1,)) if function is None else function,
        ("w",),
        ("new",),
        (torch.zeros(spec.shape, device=arena.device),),
    )
    if wrap is not None:
        cell = replace(cell, function=wrap(arena, cell.function))
    program = page_program(cell, rounds)
    candidate = analyze_region_program(program).candidates[-1]
    runner = RegionExecution(
        program,
        lower_region_schedule(program, candidate.schedule_map),
        lower_region_bindings(program, candidate.schedule_map),
        {cell.digest: cell},
    )
    tasks = tuple(Task(t.implementation_id) for t in arena.plan.schedule_physical.tasks)
    tasks[0].region, tasks[0].program = runner, program
    execution = StateExecution(
        arena,
        RequestStateDirectory(arena, arena.plan.semantic_graph.states[0]),
        Math(),
        tasks,
        AnchorId(arena.plan.timeline.anchors[-1].ordinal),
    )
    identities = (RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0))
    refs = execution.admit(identities, stream)
    rows = tuple(PackedPhaseRow(i, i, 0, HiddenPhase.PREFILL, identity) for i, identity in enumerate(identities))
    return execution, tasks[0], rows, refs


def test_rescheduled_region_publishes_many_versions_through_original_owner():
    execution, task, rows, original = owner_fixture(rounds=8)
    final = execution.step(rows)
    context = task.context
    assert len(context.page_publications) == 16
    assert not context.page_writes and all(context.page_publications.values())
    assert all(release is not None for release in context.page_pins.values())
    assert tuple(ref.version_id for ref in final) == (8, 8)
    for value, expected in zip(task.result.values, (1.0, 9.0, 10.0), strict=True):
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    pin = execution.arena.pin_exact(final)
    for ref in final:
        for field in execution.arena.snapshot(pin, ref)[1:]:
            assert torch.count_nonzero(field) == 0
    execution.arena.wait_release(execution.arena.release_pin(pin))
    # Only the selected field was privately overwritten; all other fields kept
    # the original pages. Returned old data survives retirement and pool reuse.
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0
    assert execution.arena.usage().private_versions == 0
    torch.testing.assert_close(task.result.values[0], torch.ones_like(task.result.values[0]), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="released"):
        task.region.run(
            {},
            pages=RegionResources(
                task.program,
                context,
                tuple(ResourcePageCell("layers", (i,), ref, (LayoutFieldId(0),)) for i, ref in enumerate(original)),
            ),
        )


def test_resource_aliases_and_missing_cells_rejected_before_effects():
    execution, task, rows, refs = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    for cells, message in (
        ((ResourcePageCell("layers", (0,), refs[0], (LayoutFieldId(0),)),), "cover every"),
        (tuple(ResourcePageCell("layers", (i,), refs[0], (LayoutFieldId(0),)) for i in range(2)), "alias"),
    ):
        with pytest.raises(ValueError, match=message):
            RegionResources(task.program, invocation.context, cells)
    assert not invocation.context.page_publications
    # The old request directory still excludes overlapping execution.
    with pytest.raises(RuntimeError, match="lease overlaps"):
        execution.start_step(rows)
    execution.cancel_step(invocation)
    execution.step(rows)


def test_failure_after_publish_retains_original_receipt_and_cannot_replay(monkeypatch):
    execution, task, rows, refs = owner_fixture(rounds=2)
    invocation = execution.start_step(rows)
    settle = execution.arena.settle

    def fail(receipt):
        raise RuntimeError("injected settlement failure")

    monkeypatch.setattr(execution.arena, "settle", fail)
    anchor = execution.plan.tasks[0].anchor_id
    with pytest.raises(StateSettlementError) as failed:
        execution.execute_anchor(invocation, anchor)
    receipt = failed.value.receipt
    assert receipt is invocation.context.publication
    assert receipt in invocation.context.page_publications
    assert execution.arena.latest(receipt.after_refs[0].lineage_id) == receipt.after_refs[0]
    with pytest.raises(StateSettlementError):
        execution.execute_anchor(invocation, anchor)
    assert len(invocation.context.page_publications) == 1
    monkeypatch.setattr(execution.arena, "settle", settle)
    execution.cancel_step(invocation)
    assert invocation.context.page_publications[receipt]
    assert execution.arena.usage().private_versions == 0
    execution.step(rows)  # A new invocation can acquire the original directory lease.


def test_failure_before_publish_aborts_original_private_write(monkeypatch):
    execution, task, rows, _ = owner_fixture(rounds=1)
    invocation = execution.start_step(rows)
    original = execution.arena.write_field

    def fail(*args):
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(execution.arena, "write_field", fail)
    with pytest.raises(RuntimeError, match="injected write"):
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert not invocation.context.page_writes
    assert invocation.context.release is not None
    assert execution.arena.usage().private_versions == 0
    monkeypatch.setattr(execution.arena, "write_field", original)
    execution.step(rows)


def test_foreign_publication_cannot_silently_replace_snapshot_version():
    foreign = []
    refs = []

    def wrap(arena, function):
        def changed(*args):
            if not foreign:
                # Deliberately bypass the request-directory lease to test the
                # arena's independent expected-version check at publication.
                write = arena.prepare_transition((refs[0],), fields=((LayoutFieldId(0),),), copy_fields=())
                arena.write_field(write, 0, LayoutFieldId(0)).fill_(99)
                receipt = arena.publish(write)
                arena.settle(receipt)
                foreign.append(receipt)
            return function(*args)

        return changed

    execution, task, rows, originals = owner_fixture(rounds=1, wrap=wrap)
    refs.extend(originals)
    # Use the serial candidate so the injected foreign write hits the first
    # cell whose exact snapshot has just been read.
    schedule = analyze_region_program(task.program).reference_schedule
    task.region = RegionExecution(
        task.program,
        lower_region_schedule(task.program, schedule),
        lower_region_bindings(task.program, schedule),
        task.region._cells,
    )
    invocation = execution.start_step(rows)
    with pytest.raises(RuntimeError, match="not latest"):
        execution.execute_anchor(invocation, execution.plan.tasks[0].anchor_id)
    assert not invocation.context.page_publications
    assert invocation.context.release is not None
    pin = execution.arena.pin_exact(foreign[0].after_refs)
    torch.testing.assert_close(
        execution.arena.snapshot(pin, pin.refs[0])[0],
        torch.full_like(execution.arena.snapshot(pin, pin.refs[0])[0], 99),
        rtol=0,
        atol=0,
    )
    execution.arena.wait_release(execution.arena.release_pin(pin))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_region_uses_original_stream_and_completion_receipts():
    stream = torch.cuda.Stream()
    execution, task, rows, refs = owner_fixture(rounds=8, stream=stream)
    addresses = execution.arena.pool_addresses
    final = execution.step(rows, stream)
    context = task.context
    assert all(receipt.completion is not None for receipt in context.page_publications)
    assert all(
        pin.stream is stream and release.pin is pin and release.completion is not None
        for pin, release in context.page_pins.items()
    )
    # Consumers explicitly wait for the original invocation's producer stream.
    torch.cuda.current_stream().wait_stream(stream)
    for value, expected in zip(task.result.values, (1.0, 9.0, 10.0), strict=True):
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    assert tuple(ref.version_id for ref in final) == (8, 8)
    assert execution.arena.pool_addresses == addresses
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0
    assert execution.arena.usage().pinned_versions == 0
    assert execution.arena.usage().pending_receipts == 0


def test_exported_input_views_survive_original_page_reclamation():
    execution, task, rows, _ = owner_fixture(rounds=3, function=lambda w: (w.view_as(w),))
    execution.step(rows)
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0
    # Overwrite re-admitted physical slots before inspecting the returned views.
    admitted = execution.arena.prepare_admission(("dense_fast_weights", "dense_fast_weights"))
    for root in execution.arena.writable_roots(admitted):
        root[0].fill_(99)
    for value, expected in zip(task.result.values, (1.0, 1.0, 2.0), strict=True):
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)
    execution.arena.abort(admitted)
