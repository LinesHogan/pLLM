"""End-to-end CPU oracle for the clean-room dense-page PTTS producer."""

from __future__ import annotations

import pytest
import torch

from tllm.canonical.ptts.dense_page import (
    DensePageAdmission,
    DensePageAdamWConfig,
    DensePageAdamWFieldBinding,
    DensePageStateProvider,
)
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
from tllm.contracts.state_ref import LineageId, StateRef, VersionId
from tllm.contracts.state_representation import (
    StateDeviceKind,
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


def test_cpu_dense_page_provider_consumes_signed_plans_and_preserves_pinned_v0() -> None:
    layout_request = TensorLayoutRequest(
        layouts=(
            StateTensorLayout(
                "state",
                TensorLayoutSpec(
                    namespace="test.canonical_dense_state@1",
                    fields=(
                        TensorFieldSpec(
                            LayoutFieldId(0),
                            (),
                            TensorDType.FLOAT32,
                            4,
                            MutationGranularity.PAGE,
                        ),
                        TensorFieldSpec(
                            LayoutFieldId(1),
                            (),
                            TensorDType.FLOAT32,
                            4,
                            MutationGranularity.PAGE,
                        ),
                        TensorFieldSpec(
                            LayoutFieldId(2),
                            (),
                            TensorDType.FLOAT32,
                            4,
                            MutationGranularity.PAGE,
                        ),
                        TensorFieldSpec(
                            LayoutFieldId(3),
                            (),
                            TensorDType.FLOAT32,
                            4,
                            MutationGranularity.PAGE,
                        ),
                        TensorFieldSpec(
                            LayoutFieldId(4),
                            (),
                            TensorDType.FLOAT32,
                            4,
                            MutationGranularity.PAGE,
                        ),
                        TensorFieldSpec(
                            LayoutFieldId(5),
                            (),
                            TensorDType.FLOAT32,
                            4,
                            MutationGranularity.PAGE,
                        ),
                    ),
                ),
            ),
        )
    )
    layout_decision = intern_tensor_layouts(layout_request)
    validate_tensor_layouts(layout_request, layout_decision)
    (binding,) = layout_decision.bindings
    candidate = StateProviderPlan(
        resource="state",
        provider_id="test.dense_cpu@1",
        representation=StateRepresentation.DENSE_PAGE_ARENA,
        device_kind=StateDeviceKind.CPU,
        device_ordinal=0,
        layout_id=binding.layout_id,
        layout_namespace=binding.layout_namespace,
        layout_digest=binding.layout_digest,
        max_admitted_lineages=2,
        max_live_versions_per_lineage=3,
        root_page_count=1,
        reserved_bytes=144,
        capture_stable=False,
        capabilities=("device/cpu@1", "optimizer/adamw@1"),
    )
    representation_request = StateRepresentationRequest(
        memory_budget_bytes=144,
        states=(
            StateLayoutRequirement(
                resource="state",
                layout_id=binding.layout_id,
                layout_namespace=binding.layout_namespace,
                layout_digest=binding.layout_digest,
                max_admitted_lineages=2,
                required_capabilities=("device/cpu@1", "optimizer/adamw@1"),
            ),
        ),
        candidates=(candidate,),
    )
    representation_decision = compile_state_representation(representation_request)
    validate_state_representation(representation_request, representation_decision)
    validate_state_representation_reconstruction(
        representation_request,
        representation_decision,
        (candidate,),
    )
    physical_plan = lower_state_physical_plan(
        representation_request,
        representation_decision,
        plan_epoch=1,
    )
    provider = DensePageStateProvider(
        tensor_layout_request=layout_request,
        tensor_layout_decision=layout_decision,
        representation_request=representation_request,
        representation_decision=representation_decision,
        physical_plan=physical_plan,
        local_candidates=(candidate,),
        resource="state",
    )
    assert provider.device.type == "cpu"
    assert provider.pool_bytes == 144

    first_v0 = StateRef(binding.layout_id, LineageId(1), VersionId(0))
    second_v0 = StateRef(binding.layout_id, LineageId(2), VersionId(0))
    provider.admit_batch(
        (
            DensePageAdmission(
                first_v0,
                (
                    torch.tensor(1.0),
                    torch.tensor(0.5),
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                    torch.tensor(0.0),
                ),
            ),
            DensePageAdmission(
                second_v0,
                (
                    torch.tensor(0.0),
                    torch.tensor(2.0),
                    torch.tensor(2.0),
                    torch.tensor(2.0),
                    torch.tensor(2.0),
                    torch.tensor(2.0),
                ),
            ),
        )
    )
    pinned_v0 = provider.pin_exact((first_v0,))
    old_v0 = provider.snapshot(first_v0)

    first_v1 = StateRef(binding.layout_id, LineageId(1), VersionId(1))
    second_v1 = StateRef(binding.layout_id, LineageId(2), VersionId(1))
    transition_batch = provider.prepare_transition_batch(
        (first_v0, second_v0),
        (first_v1, second_v1),
    )
    provider.compute_adamw_transition(
        transition_batch,
        DensePageAdamWFieldBinding(
            parameter=LayoutFieldId(0),
            gradient=LayoutFieldId(5),
            step=LayoutFieldId(2),
            exp_avg=LayoutFieldId(3),
            exp_avg_sq=LayoutFieldId(4),
        ),
        DensePageAdamWConfig(
            lr=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        ),
        (torch.tensor(0.25), torch.tensor(2.0)),
    )
    first_v1_root, second_v1_root = provider.publish_transition_batch(
        transition_batch
    )
    assert first_v1_root.ref == first_v1
    assert second_v1_root.ref == second_v1
    for actual, expected in zip(provider.snapshot(first_v0), old_v0, strict=True):
        torch.testing.assert_close(actual, expected)
    assert not torch.equal(provider.snapshot(first_v1)[0], old_v0[0])
    assert provider.root(first_v0).ref == first_v0

    provider.release_pin(pinned_v0)
    with pytest.raises(RuntimeError, match="unknown or reclaimed"):
        provider.root(first_v0)
    provider.retire_batch((first_v1, second_v1))
    assert provider.free_pages == 6
