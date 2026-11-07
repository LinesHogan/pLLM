"""Cold physical capacity facts for the shared CPU/CUDA PageBundle arena."""

from dataclasses import dataclass
from math import prod
from pathlib import Path

import torch

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.semantic_graph import SchedulerTimeline, SemanticGraph, StateScope
from pllm.contracts.state_representation import (
    StateDeviceKind,
    StateLayoutRequirement,
    StateProviderPlan,
    StateRepresentationRequest,
)
from pllm.contracts.tensor_layout import (
    TensorFieldSpec,
    TensorLayoutBinding,
    TensorLayoutRequest,
    TensorLayoutSpec,
)
from pllm.execution.scheduler import compile_state_schedule, intern_tensor_layouts


def compile_page_bundle_plan(
    declaration: tuple[SemanticGraph, TensorLayoutRequest],
    timeline: SchedulerTimeline,
    providers: tuple["PageBundleProviderFacts", ...],
    max_admitted_lineages: int,
    state_page_budget_bytes: int,
    device: torch.device,
) -> PlanBundle:
    """Compile original declarations once; all choices remain Rust-owned."""
    graph, layouts = declaration
    layout_decision = intern_tensor_layouts(layouts)
    candidates = tuple(
        provider.materialize_plan(
            resource=entry.resource,
            layout=entry.layout,
            binding=binding,
            max_admitted_lineages=max_admitted_lineages,
            graph=graph,
            timeline=timeline,
        )
        for entry, binding in zip(layouts.layouts, layout_decision.bindings, strict=True)
        for provider in providers
    )
    host_bytes = sum(
        max(candidate.host_bytes for candidate in candidates if candidate.resource == entry.resource)
        for entry in layouts.layouts
    )
    if host_bytes:
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        if host_bytes > available:
            raise ValueError(
                'host tables do not fit available host memory; reduce the tables or declare placement="device"'
            )
    required = (f"device/{device.type}@1", "state/exact_versions@1")
    if device.type == "cuda":
        required += ("state/fenced_reclaim@1",)
    request = StateRepresentationRequest(
        state_page_budget_bytes,
        tuple(
            StateLayoutRequirement(
                binding.resource,
                binding.layout_id,
                next(
                    candidate.max_admitted_lineages
                    for candidate in candidates
                    if candidate.resource == binding.resource
                ),
                required,
            )
            for binding in layout_decision.bindings
        ),
        candidates,
    )
    states, schedule = compile_state_schedule(graph, timeline, request)
    return PlanBundle(graph, timeline, schedule, states, layouts, layout_decision)


def page_stride_bytes(field: TensorFieldSpec) -> int:
    size = prod(field.shape) * field.dtype.itemsize
    return (size + field.alignment_bytes - 1) // field.alignment_bytes * field.alignment_bytes


def version_capacity(lineages: int, publish_lag: int, *, exact_decode: bool = False) -> int:
    """Current/private roots, retained readers and history, plus two rollover slots.

    Exact in-forward transitions retain no post-sampler history. GPU completion
    fences govern reuse, so outstanding GPU work needs no further reserved roots.
    """
    return (2 if exact_decode else 3 + publish_lag) * lineages + 2


@dataclass(frozen=True, slots=True)
class PageBundleProviderFacts:
    provider_id: str
    page_capacity: int
    device_kind: StateDeviceKind = StateDeviceKind.CPU

    def materialize_plan(
        self,
        *,
        resource: str,
        layout: TensorLayoutSpec,
        binding: TensorLayoutBinding,
        max_admitted_lineages: int,
        graph: SemanticGraph,
        timeline: SchedulerTimeline,
    ) -> StateProviderPlan:
        """Reserve versions of one instance lineage, or the provider's request capacity."""
        state = next(state for state in graph.states if state.resource == resource)
        instance = state.scope is StateScope.INSTANCE
        lag = 0 if timeline.visibility is None else timeline.visibility.lag
        capacity = min(self.page_capacity, version_capacity(1, lag)) if instance else self.page_capacity
        if instance and state.immutable:
            capacity = 1
        reserved_bytes = sum(
            capacity * page_stride_bytes(field) + max(field.alignment_bytes - field.dtype.itemsize, 0)
            for field in layout.fields
        )
        return StateProviderPlan(
            resource=resource,
            provider_id=self.provider_id,
            device_kind=self.device_kind,
            layout_id=binding.layout_id,
            max_admitted_lineages=1 if instance else max_admitted_lineages,
            page_capacity=capacity,
            root_page_count=len(layout.fields),
            reserved_bytes=0 if state.placement == "host" else reserved_bytes,
            host_bytes=reserved_bytes if state.placement == "host" else 0,
            capture_stable=self.device_kind is StateDeviceKind.CUDA,
            capabilities=tuple(
                sorted(
                    (
                        f"device/{self.device_kind.value}@1",
                        "optimizer/adamw@1",
                        "state/exact_versions@1",
                        "transition/full_root@1",
                        *(("state/fenced_reclaim@1",) if self.device_kind is StateDeviceKind.CUDA else ()),
                    )
                )
            ),
        )
