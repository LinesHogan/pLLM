"""Cold physical capacity facts for the shared CPU/CUDA PageBundle arena."""

from dataclasses import dataclass
from math import prod
import torch

from tllm.canonical.scheduler import compile_schedule, lower_physical_plan
from tllm.canonical.state_representation import compile_state_representation, lower_state_physical_plan
from tllm.canonical.tensor_layout import intern_tensor_layouts
from tllm.contracts.canonical_compilation import CanonicalPlanBundle
from tllm.contracts.semantic_graph import SemanticGraph, SchedulerTimeline
from tllm.contracts.state_representation import StateLayoutRequirement, StateRepresentationRequest
from tllm.contracts.tensor_layout import TensorLayoutRequest

from tllm.contracts.state_representation import StateDeviceKind, StateProviderPlan, StateRepresentation
from tllm.contracts.tensor_layout import TensorDType, TensorFieldSpec, TensorLayoutBinding, TensorLayoutSpec


DTYPE_BYTES = {
    TensorDType.BOOL: 1, TensorDType.UINT8: 1, TensorDType.INT32: 4, TensorDType.INT64: 8,
    TensorDType.FLOAT16: 2, TensorDType.BFLOAT16: 2, TensorDType.FLOAT32: 4, TensorDType.FLOAT64: 8,
}


def compile_page_bundle_plan(graph: SemanticGraph, timeline: SchedulerTimeline,
    layouts: TensorLayoutRequest, providers: tuple["PageBundleProviderFacts", ...],
    max_admitted_lineages: int, memory_budget_bytes: int, device: torch.device,
    *, plan_epoch: int = 1) -> CanonicalPlanBundle:
    """Compile original declarations once; all choices remain Rust-owned."""
    schedule = compile_schedule(graph, timeline)
    layout_decision = intern_tensor_layouts(layouts)
    candidates = tuple(provider.materialize_plan(resource=entry.resource, layout=entry.layout,
        binding=binding, max_admitted_lineages=max_admitted_lineages)
        for entry, binding in zip(layouts.layouts, layout_decision.bindings, strict=True)
        for provider in providers)
    required = (f"device/{device.type}@1", "state/exact_versions@1")
    if device.type == "cuda":
        required += ("state/fenced_reclaim@1",)
    request = StateRepresentationRequest(memory_budget_bytes, tuple(StateLayoutRequirement(
        binding.resource, binding.layout_id, binding.layout_namespace, binding.layout_digest,
        max_admitted_lineages, required) for binding in layout_decision.bindings), candidates)
    decision = compile_state_representation(request)
    if any(item.device_kind.value != device.type or item.device_ordinal != (device.index or 0)
        for item in decision.plans):
        raise RuntimeError("Rust-selected state placement differs from mathematical specialization")
    return CanonicalPlanBundle(graph, timeline, schedule,
        lower_physical_plan(graph, timeline, schedule, plan_epoch=plan_epoch), request, decision,
        lower_state_physical_plan(request, decision, plan_epoch=plan_epoch), layouts, layout_decision)


def page_stride_bytes(field: TensorFieldSpec) -> int:
    size = prod(field.shape) * DTYPE_BYTES[field.dtype]
    return (size + field.alignment_bytes - 1) // field.alignment_bytes * field.alignment_bytes


@dataclass(frozen=True, slots=True)
class PageBundleProviderFacts:
    provider_id: str
    max_live_versions_per_lineage: int
    device_kind: StateDeviceKind = StateDeviceKind.CPU
    device_ordinal: int = 0

    def materialize_plan(self, *, resource: str, layout: TensorLayoutSpec,
                         binding: TensorLayoutBinding, max_admitted_lineages: int) -> StateProviderPlan:
        capacity = max_admitted_lineages * self.max_live_versions_per_lineage
        reserved_bytes = sum(capacity * page_stride_bytes(field)
            + max(field.alignment_bytes - DTYPE_BYTES[field.dtype], 0) for field in layout.fields)
        return StateProviderPlan(
            resource=resource, provider_id=self.provider_id,
            representation=StateRepresentation.DENSE_PAGE_ARENA,
            device_kind=self.device_kind, device_ordinal=self.device_ordinal,
            layout_id=binding.layout_id, layout_namespace=binding.layout_namespace, layout_digest=binding.layout_digest,
            max_admitted_lineages=max_admitted_lineages,
            max_live_versions_per_lineage=self.max_live_versions_per_lineage,
            root_page_count=len(layout.fields),
            reserved_bytes=reserved_bytes,
            capture_stable=False,
            capabilities=tuple(sorted((
                f"device/{self.device_kind.value}@1", "optimizer/adamw@1", "state/exact_versions@1",
                "transition/full_root@1", *(("state/fenced_reclaim@1",) if self.device_kind is StateDeviceKind.CUDA else ()),
            ))),
        )
