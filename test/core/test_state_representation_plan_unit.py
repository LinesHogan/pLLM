"""Clean-room signed state-provider selection and projection oracle."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tllm.canonical.state_representation import (
    compile_state_representation,
    lower_state_physical_plan,
    validate_state_representation,
    validate_state_representation_reconstruction,
)
from tllm.canonical.tensor_layout import (
    intern_tensor_layouts,
    validate_tensor_layouts,
)
from tllm.contracts.state_ref import LayoutId
from tllm.contracts.state_representation import (
    StateLayoutRequirement,
    StateProviderPlan,
    StateRepresentation,
    StateRepresentationRequest,
)
from tllm.contracts.tensor_layout import (
    LayoutFieldId,
    MutationGranularity,
    StateTensorLayout,
    TensorDType,
    TensorFieldSpec,
    TensorLayoutRequest,
    TensorLayoutSpec,
)


def test_rust_selects_state_provider_and_physical_plan_only_projects() -> None:
    shared_layout = TensorLayoutSpec(
        namespace="test.low_rank@1",
        fields=(
            TensorFieldSpec(
                field_id=LayoutFieldId(0),
                shape=(4, 2),
                dtype=TensorDType.FLOAT32,
                alignment_bytes=16,
                mutation_granularity=MutationGranularity.PAGE,
            ),
        ),
    )
    distinct_layout = TensorLayoutSpec(
        namespace="test.low_rank@1",
        fields=(
            TensorFieldSpec(
                field_id=LayoutFieldId(0),
                shape=(4, 1),
                dtype=TensorDType.FLOAT32,
                alignment_bytes=16,
                mutation_granularity=MutationGranularity.PAGE,
            ),
        ),
    )
    layout_request = TensorLayoutRequest(
        layouts=(
            StateTensorLayout("aux", shared_layout),
            StateTensorLayout("other", distinct_layout),
            StateTensorLayout("weights", shared_layout),
        )
    )
    layout_decision = intern_tensor_layouts(layout_request)
    validate_tensor_layouts(layout_request, layout_decision)
    binding_by_resource = {
        binding.resource: binding for binding in layout_decision.bindings
    }
    assert (
        binding_by_resource["aux"].layout_id
        == binding_by_resource["weights"].layout_id
    )
    assert binding_by_resource["other"].layout_id != (
        binding_by_resource["weights"].layout_id
    )
    assert {
        int(binding.layout_id) for binding in layout_decision.bindings
    } == {0, 1}
    single_layout_request = TensorLayoutRequest(
        layouts=(StateTensorLayout("weights", shared_layout),)
    )
    single_layout_decision = intern_tensor_layouts(single_layout_request)
    single_binding = single_layout_decision.bindings[0]
    assert single_binding.layout_id == LayoutId(0)
    assert single_binding.layout_digest == (
        binding_by_resource["weights"].layout_digest
    )
    assert single_binding.layout_id != binding_by_resource["weights"].layout_id
    assert single_binding.layout_namespace != layout_decision.layout_namespace

    weights_layout = binding_by_resource["weights"]
    requirement = StateLayoutRequirement(
        resource="weights",
        layout_id=weights_layout.layout_id,
        layout_namespace=weights_layout.layout_namespace,
        layout_digest=weights_layout.layout_digest,
        max_admitted_lineages=2,
        required_capabilities=("optimizer/adamw@1",),
    )
    resident = StateProviderPlan(
        resource="weights",
        provider_id="a.resident@1",
        representation=StateRepresentation.RESIDENT_MODULE,
        layout_id=requirement.layout_id,
        layout_namespace=requirement.layout_namespace,
        layout_digest=requirement.layout_digest,
        max_admitted_lineages=2,
        max_live_versions_per_lineage=1,
        root_page_count=1,
        reserved_bytes=48,
        capture_stable=False,
        capabilities=("optimizer/adamw@1",),
    )
    dense = StateProviderPlan(
        resource="weights",
        provider_id="z.dense@1",
        representation=StateRepresentation.DENSE_PAGE_ARENA,
        layout_id=requirement.layout_id,
        layout_namespace=requirement.layout_namespace,
        layout_digest=requirement.layout_digest,
        max_admitted_lineages=2,
        max_live_versions_per_lineage=3,
        root_page_count=1,
        reserved_bytes=32,
        capture_stable=True,
        capabilities=("optimizer/adamw@1",),
    )
    request = StateRepresentationRequest(
        memory_budget_bytes=64,
        states=(requirement,),
        candidates=(resident, dense),
    )

    decision = compile_state_representation(request)
    validate_state_representation(request, decision)
    physical = lower_state_physical_plan(request, decision, plan_epoch=1)

    assert decision.plans == (dense,)
    assert decision.reserved_bytes == dense.reserved_bytes
    assert physical.representations == decision.plans
    assert physical.request_digest == decision.request_digest
    assert physical.representation_signature == decision.signature
    forged = replace(
        decision,
        plans=(
            replace(
                dense,
                representation=StateRepresentation.HOST_SERVICE,
                layout_id=LayoutId(999),
            ),
        ),
        signature="f" * 64,
    )
    with pytest.raises(ValueError, match="differs from Rust recomputation"):
        lower_state_physical_plan(request, forged, plan_epoch=1)
    validate_state_representation_reconstruction(
        request,
        decision,
        (resident, dense),
    )
    with pytest.raises(RuntimeError, match="parent-signed"):
        validate_state_representation_reconstruction(
            request,
            decision,
            (dense,),
        )
