"""Cold physical capacity facts for the shared CPU/CUDA PageBundle arena."""

from dataclasses import dataclass
from math import prod

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
    memory_budget_bytes: int,
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
        )
        for entry, binding in zip(layouts.layouts, layout_decision.bindings, strict=True)
        for provider in providers
    )
    required = (f"device/{device.type}@1", "state/exact_versions@1")
    if device.type == "cuda":
        required += ("state/fenced_reclaim@1",)
    request = StateRepresentationRequest(
        memory_budget_bytes,
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
    if any(
        item.device_kind.value != device.type or item.device_ordinal != (device.index or 0) for item in states.plans
    ):
        raise RuntimeError("Rust-selected state placement differs from mathematical specialization")
    return PlanBundle(graph, timeline, schedule, states, layouts, layout_decision)


def page_stride_bytes(field: TensorFieldSpec) -> int:
    size = prod(field.shape) * field.dtype.itemsize
    return (size + field.alignment_bytes - 1) // field.alignment_bytes * field.alignment_bytes


@dataclass(frozen=True, slots=True)
class PageBundleProviderFacts:
    provider_id: str
    page_capacity: int
    device_kind: StateDeviceKind = StateDeviceKind.CPU
    device_ordinal: int = 0

    def materialize_plan(
        self,
        *,
        resource: str,
        layout: TensorLayoutSpec,
        binding: TensorLayoutBinding,
        max_admitted_lineages: int,
        graph: SemanticGraph,
    ) -> StateProviderPlan:
        """Reserve one root for frozen instance state; mutable state keeps its advertised capacity."""
        state = next(state for state in graph.states if state.resource == resource)
        immutable_instance = state.scope is StateScope.INSTANCE and state.immutable
        capacity = 1 if immutable_instance else self.page_capacity
        reserved_bytes = sum(
            capacity * page_stride_bytes(field) + max(field.alignment_bytes - field.dtype.itemsize, 0)
            for field in layout.fields
        )
        return StateProviderPlan(
            resource=resource,
            provider_id=self.provider_id,
            device_kind=self.device_kind,
            device_ordinal=self.device_ordinal,
            layout_id=binding.layout_id,
            max_admitted_lineages=1 if immutable_instance else max_admitted_lineages,
            page_capacity=capacity,
            root_page_count=len(layout.fields),
            reserved_bytes=reserved_bytes,
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
