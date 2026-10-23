"""Shared builders for region and PTTS behaviour tests.

The region tests use a few representative programs (a chunk x layer wavefront,
paged layer updates) and one small request-state owner. Keeping them here lets
each test file read as scenarios instead of re-deriving fixtures.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, replace

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from pllm.authoring import BackendCompileFacts, ParameterTargetShape
from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
from pllm.contracts.region_program import (
    AffineIndex,
    BankType,
    Block,
    Call,
    Coordinate,
    Function,
    Program,
    Publish,
    Region,
    Resource,
    ScalarType,
    Scan,
    Select,
    Snapshot,
    Update,
    Value,
)
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import AnchorId, SemanticState, StateResource, StateScope
from pllm.contracts.state_representation import StateDeviceKind, StateLayoutRequirement, StateRepresentationRequest
from pllm.contracts.tensor_layout import LayoutFieldId, StateTensorLayout, TensorDType, TensorLayoutRequest
from pllm.contracts.tensor_rewrite import MatmulReassociation
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.ptts.request_state import RequestStateDirectory
from pllm.execution.region_compilation import analyze_region_program, lower_region_bindings, lower_region_schedule
from pllm.execution.region_execution import RegionExecution, RegionExecutionPlan, RegionInstructionExecution
from pllm.execution.region_resources import RegionResources, ResourcePageCell
from pllm.execution.region_storage import import_region_functions
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.region_values import Bank
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.scheduler import compile_state_schedule, intern_tensor_layouts
from pllm.execution.state_execution import StateExecution
from pllm.execution.tensor_rewrites import TensorRewriteCandidate, explore_tensor_combinations, replay_tensor_rewrites
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload

I, X, B = ScalarType.INT, ScalarType.INDEX, ScalarType.BOOL


def region(arguments, operations, yields) -> Region:
    return Region(Block(tuple(arguments), tuple(operations), tuple(yields)))


def wavefront_program(chunks, layers, element, function: Function) -> Program:
    """The author's chunk-major loop over hidden rows and layer weights; no schedule supplied."""
    weights, hidden = BankType(element, layers), BankType(element, chunks)
    layer = region(
        (Value("l", X), Value("h", element), Value("w", weights)),
        (
            Select(("w", "l"), (Value("old", element),)),
            Call(("h", "old"), (Value("next_h", element), Value("next_w", element)), function.name),
            Update(("w", "l", "next_w"), (Value("bank", weights),)),
        ),
        ("next_h", "bank"),
    )
    chunk = region(
        (Value("c", X), Value("w", weights), Value("h", hidden)),
        (
            Select(("h", "c"), (Value("input", element),)),
            Scan(("input", "w"), (Value("output", element), Value("next_w", weights)), layers, layer),
            Update(("h", "c", "output"), (Value("next_h", hidden),)),
        ),
        ("next_w", "next_h"),
    )
    return Program(
        region(
            (Value("weights", weights), Value("hidden", hidden)),
            (Scan(("weights", "hidden"), (Value("w_final", weights), Value("h_final", hidden)), chunks, chunk),),
            ("w_final", "h_final"),
        ),
        (function,),
    )


def grid_program(chunks, layers) -> Program:
    """The wavefront over integers with an opaque cell."""
    return wavefront_program(chunks, layers, I, Function("cell", (I, I), (I, I)))


def tensor_grid(chunks, layers, cell: ExportedCell) -> Program:
    """The same wavefront with a tensor cell updating hidden rows and layer weights."""
    return wavefront_program(chunks, layers, cell.declaration.inputs[0], cell.declaration)


def grid_cell(device="cpu") -> ExportedCell:
    """h' = h + w, w' = w + 1: the closed form makes every cell version checkable."""
    example = torch.ones(4, dtype=torch.float64, device=device)
    return ExportedCell.capture(lambda h, w: (h + w, w + 1), ("h", "w"), ("nh", "nw"), (example, example.clone()))


def inputs_for(chunks, layers, device="cpu") -> dict[str, Bank]:
    return {
        "weights": Bank(tuple(torch.full((4,), l + 1.0, dtype=torch.float64, device=device) for l in range(layers))),
        "hidden": Bank(tuple(torch.full((4,), c + 2.0, dtype=torch.float64, device=device) for c in range(chunks))),
    }


def assert_grid_closed_form(values, chunks, layers) -> None:
    """Independent oracle for ``grid_cell``: chunk c reads layer l at its c-th version."""
    weights, hidden = values
    for l, value in enumerate(weights.elements):
        torch.testing.assert_close(value, torch.full_like(value, l + 1 + chunks), rtol=0, atol=0)
    for c, value in enumerate(hidden.elements):
        expected = c + 2 + layers * c + layers * (layers + 1) / 2
        torch.testing.assert_close(value, torch.full_like(value, expected), rtol=0, atol=0)


def bind(program: Program, cells=None, schedule: str | None = None) -> RegionExecution:
    """Bind the most optimized source candidate unless a schedule is given."""
    if schedule is None:
        schedule = analyze_region_program(program).candidates[-1].schedule_map
    return RegionExecution(
        program, lower_region_schedule(program, schedule), lower_region_bindings(program, schedule), cells or {}
    )


def single_call(cell) -> Program:
    declaration = cell.declaration
    inputs = tuple(Value(f"in{i}", t) for i, t in enumerate(declaration.inputs))
    outputs = tuple(Value(f"out{i}", t) for i, t in enumerate(declaration.outputs))
    return Program(
        region(
            inputs, (Call(tuple(v.name for v in inputs), outputs, declaration.name),), tuple(v.name for v in outputs)
        ),
        (declaration,),
    )


class StorageObserver(TorchDispatchMode):
    """Observe live real tensor storages, excluding allocator and kernel scratch."""

    def __init__(self, inputs):
        super().__init__()
        self.tensors = [weakref.ref(t) for t in inputs]
        self.peak = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        outputs = result if isinstance(result, tuple) else (result,)
        self.tensors.extend(weakref.ref(t) for t in outputs if isinstance(t, torch.Tensor))
        self.tensors = [ref for ref in self.tensors if ref() is not None]
        alive = [ref() for ref in self.tensors]
        storages = {
            (str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
            for t in alive
            if t is not None and t.untyped_storage().nbytes()
        }
        self.peak = max(self.peak, sum(storages.values()))
        return result


def reassociation_candidates(source: ExportedCell) -> tuple[TensorRewriteCandidate[MatmulReassociation], ...]:
    """Single-step matmul reassociation candidates through the general explore/replay door."""
    return tuple(
        TensorRewriteCandidate(replay_tensor_rewrites(source, p), p[0])
        for p in explore_tensor_combinations((source,), max_depth=1)
        if isinstance(p[0], MatmulReassociation)
    )


def stream_execution(program, cells, plan, *, payload_limits):
    return RegionExecutionPlan(
        program, import_region_functions(program, cells), RegionScheduleProposal(plan.payload, plan), payload_limits
    ).bind(cells)


def limits(plan) -> tuple[RegionPayloadLimit, ...]:
    return tuple(RegionPayloadLimit(d.device, d.peak_bytes_upper_bound) for d in plan.devices)


def paged_rows(positions=(4, 5, 4, 5)) -> tuple[PackedPhaseRow, ...]:
    """Four prefill rows from two requests, two tokens each."""
    return tuple(
        PackedPhaseRow(i, i // 2, position, HiddenPhase.PREFILL, RequestIdentity(str(i // 2), i // 2, 0))
        for i, position in enumerate(positions)
    )


def inplace_plan(device: torch.device, provider: PageBundleProviderFacts, capabilities: tuple[str, ...]):
    """The InPlaceTTT plan for one request-scoped dense resource on ``device``."""
    target = ParameterTarget(0, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    probe = BackendPlacementProbe(
        provider.provider_id,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        AttachmentTopology.IN_PROCESS,
        AttachmentGraphMode.EAGER,
        provider.device_kind,
        0,
        1,
        1,
        TensorDType.FLOAT32,
        3,
        8,
        (),
        (),
    )
    timeline = RuntimeFacts(device).timeline(capabilities)
    facts = BackendCompileFacts(probe, timeline, (provider,), 2, 4096, 0, (ParameterTargetShape(target, 3, 2),))
    return InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=2)).plan_backend(facts).inputs.plan


def page_arena(*, heterogeneous: bool = False) -> PageBundleArena:
    """A CPU arena for one request-scoped dense resource (plus an instance one)."""
    provider = PageBundleProviderFacts("test.page_bundle@1", 6)
    plan = inplace_plan(torch.device("cpu"), provider, ("transition/full_root@1",))
    if heterogeneous:
        (dense,) = plan.tensor_layout_request.layouts
        layout = replace(
            dense.layout, namespace="test.global_factors@1", fields=(replace(dense.layout.fields[0], shape=(2, 5)),)
        )
        layouts = TensorLayoutRequest((dense, StateTensorLayout("global_factors", layout)))
        graph = replace(
            plan.semantic_graph,
            states=(
                *plan.semantic_graph.states,
                SemanticState(StateResource("global_factors"), StateScope.INSTANCE, ()),
            ),
        )
        plan = replan_states(plan, graph, layouts, 4096)
    return PageBundleArena(plan)


def replan_states(plan, graph, layouts: TensorLayoutRequest, memory_bytes: int):
    """Plan a changed state set through the ordinary Rust state scheduler."""
    bindings = intern_tensor_layouts(layouts)
    provider = PageBundleProviderFacts("test.page_bundle@1", 6)
    candidates = tuple(
        provider.materialize_plan(
            resource=entry.resource, layout=entry.layout, binding=binding, max_admitted_lineages=2
        )
        for entry, binding in zip(layouts.layouts, bindings.bindings, strict=True)
    )
    request = StateRepresentationRequest(
        memory_bytes,
        tuple(
            StateLayoutRequirement(b.resource, b.layout_id, b.layout_namespace, b.layout_digest, 2)
            for b in bindings.bindings
        ),
        candidates,
    )
    decision, physical, schedule, scheduled = compile_state_schedule(graph, plan.timeline, request)
    return replace(
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


def page_program(cell: ExportedCell, rounds=8) -> Program:
    """`rounds` chunks each update both layer pages; snapshots before and after."""
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


class PageMath:
    """Initialize request i's first field to i+1 and report owned refs."""

    parameter_phases = ()

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
class PageTask:
    """Runs ``region`` over the invocation's two request pages."""

    implementation_id: int
    region: RegionInstructionExecution | None = None
    streams: RegionStreamBindings | None = None
    program: Program | None = None
    result = None
    context = None
    inputs = None

    def execute(self, context):
        if self.region is not None:
            self.context = context
            self.result = self.region.run(
                {} if self.inputs is None else self.inputs, pages=self.pages(context), streams=self.streams
            )

    def pages(self, context, refs=None) -> RegionResources:
        refs = context.pin.refs if refs is None else refs
        cells = tuple(ResourcePageCell("layers", (i,), ref, (LayoutFieldId(0),)) for i, ref in enumerate(refs))
        return RegionResources(self.program, context, cells)


def cuda_arena(stream) -> PageBundleArena:
    """The same request plan, with pages and completion receipts on CUDA."""
    provider = PageBundleProviderFacts("test.region.cuda@1", 6, StateDeviceKind.CUDA)
    return PageBundleArena(inplace_plan(stream.device, provider, ("transition/full_root@1", "state/fenced_reclaim@1")))


def owner_fixture(*, rounds=8, stream=None):
    """Two admitted requests whose first task runs ``page_program`` on their pages."""
    arena = page_arena() if stream is None else cuda_arena(stream)
    spec = arena.plan.tensor_layout_request.layouts[0].layout.fields[0]
    cell = ExportedCell.capture(lambda w: (w + 1,), ("w",), ("new",), (torch.zeros(spec.shape, device=arena.device),))
    program = page_program(cell, rounds)
    tasks = tuple(PageTask(t.implementation_id) for t in arena.plan.schedule_physical.tasks)
    tasks[0].region, tasks[0].program = bind(program, {cell.digest: cell}), program
    execution = StateExecution(
        arena,
        RequestStateDirectory(arena, arena.plan.semantic_graph.states[0]),
        PageMath(),
        tasks,
        AnchorId(arena.plan.timeline.anchors[-1].ordinal),
    )
    identities = (RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0))
    refs = execution.admit(identities, stream)
    rows = tuple(PackedPhaseRow(i, i, 0, HiddenPhase.PREFILL, identity) for i, identity in enumerate(identities))
    return execution, tasks[0], rows, refs


def retire_all(execution, rows) -> None:
    """Retire both requests and require every page and pin to be reclaimed."""
    execution.retire(tuple(row.identity for row in rows))
    execution.arena.collect_completed(wait=True)
    assert execution.arena.active_lineages == 0
    usage = execution.arena.usage()
    assert usage.private_versions == usage.pinned_versions == usage.pending_receipts == 0
