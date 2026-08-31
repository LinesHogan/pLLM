"""Shared signed-plan validation for canonical dense-page providers."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.canonical.state_representation import (
    validate_state_physical_plan,
    validate_state_representation,
    validate_state_representation_reconstruction,
)
from tllm.canonical.tensor_layout import validate_tensor_layouts
from tllm.contracts.state_representation import (
    StateDeviceKind,
    StatePhysicalPlan,
    StateProviderPlan,
    StateRepresentation,
    StateRepresentationDecision,
    StateRepresentationRequest,
)
from tllm.contracts.tensor_layout import (
    StateTensorLayout,
    TensorLayoutDecision,
    TensorLayoutRequest,
    TensorLayoutSpec,
)


@dataclass(frozen=True, slots=True)
class DensePageSignedSpec:
    layout: TensorLayoutSpec
    selected_plan: StateProviderPlan
    physical_rows: int


def validate_dense_page_provider(
    *,
    tensor_layout_request: TensorLayoutRequest,
    tensor_layout_decision: TensorLayoutDecision,
    representation_request: StateRepresentationRequest,
    representation_decision: StateRepresentationDecision,
    physical_plan: StatePhysicalPlan,
    local_candidates: tuple[StateProviderPlan, ...],
    resource: str,
    device_kind: StateDeviceKind,
    device_ordinal: int,
) -> DensePageSignedSpec:
    validate_tensor_layouts(tensor_layout_request, tensor_layout_decision)
    validate_state_representation(representation_request, representation_decision)
    validate_state_representation_reconstruction(
        representation_request,
        representation_decision,
        local_candidates,
    )
    validate_state_physical_plan(
        representation_request,
        representation_decision,
        physical_plan,
    )
    if (
        physical_plan.request_digest != representation_decision.request_digest
        or physical_plan.representation_signature
        != representation_decision.signature
        or physical_plan.representations != representation_decision.plans
        or physical_plan.reserved_bytes != representation_decision.reserved_bytes
    ):
        raise RuntimeError("physical state plan differs from its signed decision")
    if type(resource) is not str or not resource:
        raise ValueError("dense-page provider requires a resource")
    layout_items = tuple(
        item for item in tensor_layout_request.layouts if item.resource == resource
    )
    layout_bindings = tuple(
        binding
        for binding in tensor_layout_decision.bindings
        if binding.resource == resource
    )
    selected_plans = tuple(
        plan
        for plan in representation_decision.plans
        if plan.resource == resource
    )
    requirements = tuple(
        requirement
        for requirement in representation_request.states
        if requirement.resource == resource
    )
    if (
        len(layout_items) != 1
        or len(layout_bindings) != 1
        or len(selected_plans) != 1
        or len(requirements) != 1
    ):
        raise RuntimeError("signed plans do not define one selected state resource")
    layout_item: StateTensorLayout = layout_items[0]
    layout_binding = layout_bindings[0]
    selected = selected_plans[0]
    requirement = requirements[0]
    if (
        selected.representation is not StateRepresentation.DENSE_PAGE_ARENA
        or selected.device_kind is not device_kind
        or selected.device_ordinal != device_ordinal
        or selected.layout_id != layout_binding.layout_id
        or selected.layout_namespace != layout_binding.layout_namespace
        or selected.layout_digest != layout_binding.layout_digest
        or selected.max_admitted_lineages
        != requirement.max_admitted_lineages
        or selected.max_live_versions_per_lineage < 2
        or selected.root_page_count != 1
    ):
        raise RuntimeError(
            "signed dense provider differs from layout or placement facts"
        )
    return DensePageSignedSpec(
        layout_item.layout,
        selected,
        selected.max_admitted_lineages
        * selected.max_live_versions_per_lineage,
    )


__all__ = ["DensePageSignedSpec", "validate_dense_page_provider"]
