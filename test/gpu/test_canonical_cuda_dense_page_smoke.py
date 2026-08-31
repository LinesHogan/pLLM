"""One real-CUDA fence/COW proof for the canonical dense-page provider."""

from __future__ import annotations

import pytest
import torch

from tllm.canonical.ptts import (
    CudaDensePageAdmission,
    CudaDensePageStateProvider,
    DensePageAdamWConfig,
    DensePageAdamWFieldBinding,
    DensePageCudaProviderFacts,
)
from tllm.canonical.state_representation import (
    compile_state_representation,
    lower_state_physical_plan,
    validate_state_representation,
    validate_state_representation_reconstruction,
)
from tllm.canonical.tensor_layout import intern_tensor_layouts
from tllm.contracts.state_ref import LineageId, StateRef, VersionId
from tllm.contracts.state_representation import (
    StateLayoutRequirement,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_dense_page_cow_waits_for_exact_completion_fence() -> None:
    layout = TensorLayoutSpec(
        namespace="test.canonical_cuda_dense@1",
        fields=(
            TensorFieldSpec(
                LayoutFieldId(0),
                (4,),
                TensorDType.FLOAT32,
                16,
                MutationGranularity.PAGE,
            ),
            TensorFieldSpec(
                LayoutFieldId(1),
                (4,),
                TensorDType.FLOAT32,
                16,
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
                (4,),
                TensorDType.FLOAT32,
                16,
                MutationGranularity.PAGE,
            ),
            TensorFieldSpec(
                LayoutFieldId(4),
                (4,),
                TensorDType.FLOAT32,
                16,
                MutationGranularity.PAGE,
            ),
        ),
    )
    layout_request = TensorLayoutRequest(
        (StateTensorLayout("state", layout),)
    )
    layout_decision = intern_tensor_layouts(layout_request)
    (binding,) = layout_decision.bindings
    candidate = DensePageCudaProviderFacts(
        "canonical.cuda_dense_page@1",
        device_ordinal=0,
        max_live_versions_per_lineage=3,
    ).materialize_plan(
        resource="state",
        layout=layout,
        binding=binding,
        max_admitted_lineages=2,
    )
    representation_request = StateRepresentationRequest(
        memory_budget_bytes=candidate.reserved_bytes,
        states=(
            StateLayoutRequirement(
                resource="state",
                layout_id=binding.layout_id,
                layout_namespace=binding.layout_namespace,
                layout_digest=binding.layout_digest,
                max_admitted_lineages=2,
                required_capabilities=(
                    "device/cuda@1",
                    "optimizer/adamw@1",
                    "state/exact_versions@1",
                    "state/fenced_reclaim@1",
                ),
            ),
        ),
        candidates=(candidate,),
    )
    representation = compile_state_representation(representation_request)
    validate_state_representation(representation_request, representation)
    validate_state_representation_reconstruction(
        representation_request,
        representation,
        (candidate,),
    )
    physical = lower_state_physical_plan(
        representation_request,
        representation,
        plan_epoch=1,
    )
    provider = CudaDensePageStateProvider(
        tensor_layout_request=layout_request,
        tensor_layout_decision=layout_decision,
        representation_request=representation_request,
        representation_decision=representation,
        physical_plan=physical,
        local_candidates=(candidate,),
        resource="state",
        device_ordinal=0,
    )
    device = torch.device("cuda", 0)
    writer = torch.cuda.Stream(device=device)
    reader = torch.cuda.Stream(device=device)
    first_v0 = StateRef(binding.layout_id, LineageId(1), VersionId(0))
    second_v0 = StateRef(binding.layout_id, LineageId(2), VersionId(0))
    with torch.cuda.stream(writer):
        initial_values = (
            torch.tensor((1.0, 2.0, 3.0, 4.0), device=device),
            torch.tensor((-1.0, 0.5, 2.0, -0.5), device=device),
        )
    admissions: list[CudaDensePageAdmission] = []
    for ref, parameter in zip(
        (first_v0, second_v0),
        initial_values,
        strict=True,
    ):
        initial = provider.new_zero_root(writer)
        with torch.cuda.stream(writer):
            initial.tensors[0].copy_(parameter)
        admissions.append(CudaDensePageAdmission(ref, initial))
    prepared = provider.prepare_admission_batch(tuple(admissions))
    provider.abort_unsubmitted_admission(prepared)
    assert provider.free_pages == 6

    prepared = provider.prepare_admission_batch(tuple(admissions))
    with pytest.raises(RuntimeError, match="another initialization stream"):
        provider.submit_admission_batch(prepared, reader)
    provider.submit_admission_batch(prepared, writer)
    pinned_v0 = provider.pin_exact((first_v0, second_v0), reader)
    with pytest.raises(RuntimeError, match="another consumer stream"):
        provider.snapshot(pinned_v0, first_v0, writer)
    original_v0 = provider.snapshot(pinned_v0, first_v0, reader)

    first_v1 = StateRef(binding.layout_id, LineageId(1), VersionId(1))
    second_v1 = StateRef(binding.layout_id, LineageId(2), VersionId(1))
    transition = provider.prepare_transition_batch(
        (first_v0, second_v0),
        (first_v1, second_v1),
        writer,
    )
    adamw_fields = DensePageAdamWFieldBinding(
        parameter=LayoutFieldId(0),
        gradient=LayoutFieldId(1),
        step=LayoutFieldId(2),
        exp_avg=LayoutFieldId(3),
        exp_avg_sq=LayoutFieldId(4),
    )
    adamw_config = DensePageAdamWConfig(
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    with torch.cuda.stream(writer):
        gradients = (
            torch.full((4,), 0.25, device=device),
            torch.full((4,), -0.5, device=device),
        )
    bound_gradients = provider.bind_gradients(gradients, writer)
    with pytest.raises(RuntimeError, match="another prepared stream"):
        provider.submit_adamw_transition(
            transition,
            adamw_fields,
            adamw_config,
            bound_gradients,
            reader,
        )
    publication_v1 = provider.submit_adamw_transition(
        transition,
        adamw_fields,
        adamw_config,
        bound_gradients,
        writer,
    )
    assert publication_v1.refs == (first_v1, second_v1)
    assert provider.free_pages == 2
    v0_release = provider.release_pin(pinned_v0, reader)
    assert provider.pending_pin_releases == 1
    with pytest.raises(RuntimeError, match="no new reads may start"):
        provider.snapshot(pinned_v0, first_v0, reader)
    provider.wait_release(v0_release)
    provider.require_stable()

    pinned_v1 = provider.pin_exact((first_v1, second_v1), reader)
    first_v1_snapshot = provider.snapshot(pinned_v1, first_v1, reader)
    v1_release = provider.release_pin(pinned_v1, reader)
    provider.wait_release(v1_release)
    provider.collect_completed()
    assert provider.free_pages == 4
    torch.testing.assert_close(original_v0[0], initial_values[0])
    assert not torch.equal(first_v1_snapshot[0], original_v0[0])
    provider.retire_batch((first_v1, second_v1))
    assert provider.free_pages == 6
