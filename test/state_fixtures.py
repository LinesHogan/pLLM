"""Independent storage and page fixtures for state lifecycle and route tests."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, replace

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from pllm import testing
from pllm.config import ExecutionOptions
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import Projection, ProjectionShape
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.semantic_graph import AnchorId, SemanticState, StateResource, StateScope
from pllm.contracts.state_representation import StateDeviceKind, StateRepresentationRequest
from pllm.contracts.tensor_layout import LayoutFieldId, StateTensorLayout, TensorLayoutRequest
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.scheduler import compile_state_schedule, intern_tensor_layouts
from pllm.execution.state_execution import MathBinding
from pllm.execution.tensor_program import FunctionMath
from test.dense_update import DenseUpdate, DenseUpdateConfig


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


def paged_rows(positions=(4, 5, 4, 5)) -> tuple[PackedPhaseRow, ...]:
    """Four prefill rows from two requests, two tokens each."""
    return tuple(
        PackedPhaseRow(i, i // 2, position, HiddenPhase.PREFILL, RequestIdentity(str(i // 2), i // 2, 0))
        for i, position in enumerate(positions)
    )


def inplace_plan(device: torch.device, provider: PageBundleProviderFacts, capabilities: tuple[str, ...]):
    """The InPlaceTTT plan for one request-scoped dense resource on ``device``."""
    target = Projection("down_proj", 0)
    timeline = model_timeline(device, capabilities)
    facts = testing.backend_facts(
        3, (), timeline, provider, memory_bytes=4096, max_rows=8, targets=(ProjectionShape(target, 3, 2),)
    )
    workload = DenseUpdate(DenseUpdateConfig((target,), chunk_size=2))
    return workload.plan_backend(*workload.specialize(facts, ExecutionOptions())).plan


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
            resource=entry.resource,
            layout=entry.layout,
            binding=binding,
            max_admitted_lineages=2,
            graph=graph,
            timeline=plan.timeline,
        )
        for entry, binding in zip(layouts.layouts, bindings.bindings, strict=True)
    )
    request = StateRepresentationRequest(
        memory_bytes,
        tuple(p.resource for p in candidates),
        candidates,
    )
    states, schedule = compile_state_schedule(graph, plan.timeline, request)
    return replace(
        plan,
        semantic_graph=graph,
        schedule=schedule,
        states=states,
        tensor_layout_request=layouts,
        tensor_layout_decision=bindings,
    )


def cuda_arena(stream) -> PageBundleArena:
    """The same request plan, with pages and completion receipts on CUDA."""
    provider = PageBundleProviderFacts("test.region.cuda@1", 6, StateDeviceKind.CUDA)
    return PageBundleArena(inplace_plan(stream.device, provider, ("transition/full_root@1", "state/fenced_reclaim@1")))


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
    """Run direct tensor publications through the invocation's state owner."""

    implementation_id: int
    update = None
    context = None

    def execute(self, context):
        """Run the configured update in this invocation."""
        if self.update is not None:
            self.context = context
            self.update(context)
        return {}


def increment_resources(context):
    """Publish one increment of the first field of every selected resource."""
    refs = context.current_owned_refs(context.pin.refs)
    pin = context.arena.pin_exact(refs, context.pin.stream)
    context.register_pin(pin)
    for ref in refs:
        (value,) = context.arena.snapshot(pin, ref)[:1]
        write = context.prepare_owned_write(StateRefs.of((ref,)), ((LayoutFieldId(0),),))
        context.arena.write_field(write, 0, LayoutFieldId(0)).copy_(value + 1)
        context.publish_owned_write(write)
    context.release_owned_pin(pin)


def multistate(*, route=False, instance=True, failing=()):
    """Bind heterogeneous state with optional route sharing and admission failure."""
    arena = multistate_arena(route=route, instance=instance)

    def initializer(name, value):
        def initialize(roots, keys):
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
        lambda step: step.rows.rows,
        lambda step, batch: step.rows.rows,
        lambda rows, output: output.selection,
    )
    tasks = tuple(ResourceTask(task.implementation_id) for task in arena.plan.schedule.tasks)
    execution = MathBinding(functions, tasks).bind(arena, AnchorId(arena.plan.timeline.anchors[-1].ordinal))
    return execution, tasks[0]


def step_for(*identities):
    """A step with one prefill row per request identity."""
    return testing.step_input(identities, tokens=(HiddenPhase.PREFILL,) * len(identities), sampler_order=())


def assert_multistate_reclaimed(execution):
    """Require retirement to reclaim every resource, inference pin and receipt."""
    execution.arena.collect_completed(wait=True)
    usage = execution.arena.usage()
    assert execution.directory.active_count == execution.arena.active_states == 0
    assert execution.directory.inference_pin_count == 0
    assert usage.private_versions == usage.pinned_versions == usage.pending_receipts == 0
