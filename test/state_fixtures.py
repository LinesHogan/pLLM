"""Independent storage and page fixtures for state lifecycle and route tests."""

from __future__ import annotations

import weakref
from dataclasses import replace

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from pllm.contracts.backend_attachment import AttachmentGraphMode, AttachmentTopology, BackendPlacementProbe
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget, ParameterTargetShape
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import SemanticState, StateResource, StateScope
from pllm.contracts.state_representation import StateDeviceKind, StateLayoutRequirement, StateRepresentationRequest
from pllm.contracts.tensor_layout import StateTensorLayout, TensorDType, TensorLayoutRequest
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.scheduler import compile_state_schedule, intern_tensor_layouts
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload


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
    facts = BackendCompileFacts(
        replace(probe, targets=(ParameterTargetShape(target, 3, 2),)), timeline, (provider,), 2, 4096, 0
    )
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
            resource=entry.resource, layout=entry.layout, binding=binding, max_admitted_lineages=2, graph=graph
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


def cuda_arena(stream) -> PageBundleArena:
    """The same request plan, with pages and completion receipts on CUDA."""
    provider = PageBundleProviderFacts("test.region.cuda@1", 6, StateDeviceKind.CUDA)
    return PageBundleArena(inplace_plan(stream.device, provider, ("transition/full_root@1", "state/fenced_reclaim@1")))
