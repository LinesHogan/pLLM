"""One end-to-end CPU boundary proof for canonical ESamp request state."""

from __future__ import annotations

import pytest
import torch

from tllm.canonical.kernels.low_rank_gated import (
    LowRankGatedLayout,
    low_rank_gated_forward,
)
from tllm.canonical.ptts import DensePageCpuProviderFacts
from tllm.canonical.workloads.esamp_request import ESampRequestWorkload
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestCompileOptions,
    ESampRequestConfig,
    ESampRequestPackedHiddenBindings,
    ESampRequestSpecialization,
    ESampRequestStepInput,
    ESAMP_PARAMETER_FIELD,
)
from tllm.contracts.hidden_locator import (
    HiddenLayerKind,
    HiddenPhase,
    HiddenSite,
)
from tllm.contracts.packed_hidden import (
    PackedHiddenResourceBinding,
    PackedHiddenSnapshot,
    PackedHiddenSnapshotBatch,
)
from tllm.contracts.packed_rows import (
    PackedPhaseRow,
    PackedPhaseRowBatch,
    PackedSamplerRowBatch,
    PackedSamplerRowLink,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    SchedulerTimeline,
    TimelineAnchor,
)
from tllm.contracts.state_representation import StateRepresentation
from tllm.contracts.tensor_layout import TensorDType


def test_esamp_request_executes_rust_ordered_ptts_version_lifecycle() -> None:
    state_budget_bytes = 4000
    identities = (
        RequestIdentity("request-a", 0, 0),
        RequestIdentity("request-b", 1, 0),
    )
    hidden_bindings = ESampRequestPackedHiddenBindings(
        source=PackedHiddenResourceBinding(
            "0",
            HiddenLayerKind.INDEX,
            HiddenSite.BLOCK_OUTPUT,
            "model.layers[0]",
        ),
        target=PackedHiddenResourceBinding(
            "-1",
            HiddenLayerKind.INDEX,
            HiddenSite.BLOCK_OUTPUT,
            "model.layers[-1]",
        ),
    )
    workload = ESampRequestWorkload(
        ESampRequestConfig(
            rank=2,
            intervention=True,
            intervention_beta=0.2,
        )
    )
    execution = workload.compile(
        ESampRequestSpecialization(
            hidden_size=4,
            tensor_dtype=TensorDType.FLOAT32,
            initialization_seed=7,
            hidden_bindings=hidden_bindings,
        ),
        ESampRequestCompileOptions(
            timeline=SchedulerTimeline(
                anchors=(
                    TimelineAnchor(AnchorSymbol("model.execute"), 0),
                    TimelineAnchor(AnchorSymbol("sampler.pre_draw"), 1),
                    TimelineAnchor(AnchorSymbol("token.commit"), 2),
                    TimelineAnchor(AnchorSymbol("next.prepare"), 3),
                ),
                capabilities=(
                    "compute/cpu@1",
                    "state/dense_page_cpu@1",
                ),
            ),
            providers=(
                DensePageCpuProviderFacts(
                    "canonical.cpu_dense_page@1",
                    max_live_versions_per_lineage=3,
                ),
            ),
            max_admitted_lineages=2,
            state_memory_budget_bytes=state_budget_bytes,
        ),
    )
    assert tuple(int(task.task_id) for task in execution.plan.tasks) == (1, 0, 2)
    assert execution.plan.fallback_count == 0
    assert 0 < execution.state_plan.reserved_bytes <= state_budget_bytes
    assert (
        execution.state_plan.representations[0].representation
        is StateRepresentation.DENSE_PAGE_ARENA
    )

    version_zero = execution.admit(identities)
    version_zero_parameters = tuple(
        execution.snapshot(ref)[int(ESAMP_PARAMETER_FIELD)] for ref in version_zero
    )
    row_values = (
        PackedPhaseRow(
            0,
            0,
            "request-a",
            0,
            HiddenPhase.DECODE,
            identities[0],
            0,
        ),
        PackedPhaseRow(
            1,
            1,
            "request-b",
            0,
            HiddenPhase.DECODE,
            identities[1],
            1,
        ),
    )
    sampler_links = (
        PackedSamplerRowLink(0, 0),
        PackedSamplerRowLink(1, 1),
    )
    source = torch.tensor(
        ((0.2, -0.1, 0.4, 0.3), (-0.5, 0.7, 0.1, -0.2)),
        dtype=torch.float32,
    )
    target = torch.tensor(
        ((0.6, 0.0, -0.3, 0.2), (0.4, -0.2, 0.8, 0.1)),
        dtype=torch.float32,
    )
    hidden_by_resource = {
        hidden_bindings.source: source,
        hidden_bindings.target: target,
    }
    first_rows = PackedPhaseRowBatch(1, 0, 2, row_values)
    first_handle = execution.start_step(
        ESampRequestStepInput(
            first_rows,
            PackedSamplerRowBatch(first_rows, sampler_links),
            PackedHiddenSnapshotBatch(
                first_rows,
                tuple(
                    PackedHiddenSnapshot(
                        resource,
                        hidden_by_resource[resource],
                    )
                    for resource in hidden_bindings.resources
                ),
            ),
        )
    )
    execution.execute_anchor(first_handle, AnchorId(0))
    assert execution.sampler_guidance(first_handle).affected_row_ids == ()
    assert execution.diagnostics().update_count == 0
    with pytest.raises(RuntimeError, match="before its signed anchor"):
        execution.complete_sampler(first_handle)
    execution.execute_anchor(first_handle, AnchorId(1))
    execution.complete_sampler(first_handle)
    execution.execute_anchor(first_handle, AnchorId(2))
    for ref, expected in zip(
        version_zero,
        version_zero_parameters,
        strict=True,
    ):
        torch.testing.assert_close(
            execution.snapshot(ref)[int(ESAMP_PARAMETER_FIELD)],
            expected,
        )
    first = execution.finish_step(first_handle)
    assert tuple(int(ref.version_id) for ref in first.before_refs) == (0, 0)
    assert tuple(int(ref.version_id) for ref in first.after_refs) == (1, 1)
    assert all(float(loss) > 0.0 for loss in first.losses)
    with pytest.raises(RuntimeError, match="unknown or reclaimed"):
        execution.snapshot(version_zero[0])

    second_rows = PackedPhaseRowBatch(1, 1, 2, row_values)
    second_handle = execution.start_step(
        ESampRequestStepInput(
            second_rows,
            PackedSamplerRowBatch(second_rows, sampler_links),
            PackedHiddenSnapshotBatch(
                second_rows,
                tuple(
                    PackedHiddenSnapshot(
                        resource,
                        hidden_by_resource[resource],
                    )
                    for resource in hidden_bindings.resources
                ),
            ),
        )
    )
    execution.execute_anchor(second_handle, AnchorId(0))
    second_guidance = execution.sampler_guidance(second_handle)
    assert second_guidance.affected_row_ids == (0, 1)
    pinned_parameters = torch.stack(
        tuple(
            execution.snapshot(ref)[int(ESAMP_PARAMETER_FIELD)]
            for ref in first.after_refs
        )
    )
    expected_guidance = low_rank_gated_forward(
        source,
        pinned_parameters,
        torch.tensor((0, 1), dtype=torch.long),
        LowRankGatedLayout(4, 2, True),
    )
    torch.testing.assert_close(second_guidance.pred_hidden, expected_guidance)
    execution.execute_anchor(second_handle, AnchorId(1))
    execution.complete_sampler(second_handle)
    execution.execute_anchor(second_handle, AnchorId(2))
    second = execution.finish_step(second_handle)
    assert tuple(int(ref.version_id) for ref in second.before_refs) == (1, 1)
    assert tuple(int(ref.version_id) for ref in second.after_refs) == (2, 2)
    assert second.guidance.affected_row_ids == (0, 1)
    assert second.sampler_beta == 0.2
    third_rows = PackedPhaseRowBatch(1, 2, 2, row_values)
    execution.start_step(
        ESampRequestStepInput(
            third_rows,
            PackedSamplerRowBatch(third_rows, sampler_links),
            PackedHiddenSnapshotBatch(
                third_rows,
                tuple(
                    PackedHiddenSnapshot(
                        resource,
                        hidden_by_resource[resource],
                    )
                    for resource in hidden_bindings.resources
                ),
            ),
        )
    )
    execution.retire(identities)
    diagnostics = execution.diagnostics()
    assert diagnostics.active_lineages == 0
    assert diagnostics.loss_count == 4
    assert diagnostics.update_count == 4
    assert diagnostics.completed_sampler_pins == 2
    assert execution.free_state_pages == 6
