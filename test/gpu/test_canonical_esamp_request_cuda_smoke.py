"""One real-CUDA vertical proof for canonical ESamp request execution."""

from __future__ import annotations

import pytest
import torch

from tllm.canonical.cuda_packed_hidden import CudaPackedHiddenSnapshotBatch
from tllm.canonical.ptts import DensePageCudaProviderFacts
from tllm.canonical.workloads.esamp_request import ESampRequestWorkload
from tllm.canonical.workloads.esamp_request_cuda import (
    ESampRequestCudaExecution,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestCompileOptions,
    ESampRequestConfig,
    ESampRequestPackedHiddenBindings,
    ESampRequestSpecialization,
    ESampRequestStepInput,
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
from tllm.contracts.state_representation import StateDeviceKind
from tllm.contracts.tensor_layout import TensorDType


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_esamp_request_runs_packed_guidance_and_versioned_training() -> None:
    device = torch.device("cuda", 0)
    execution_stream = torch.cuda.Stream(device=device)
    sampler_stream = torch.cuda.Stream(device=device)
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
    compiled = workload.compile(
        ESampRequestSpecialization(
            hidden_size=4,
            tensor_dtype=TensorDType.FLOAT32,
            initialization_seed=7,
            hidden_bindings=hidden_bindings,
            device_kind=StateDeviceKind.CUDA,
            device_ordinal=0,
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
                    "compute/cuda@1",
                    "state/dense_page_cuda@1",
                ),
            ),
            providers=(
                DensePageCudaProviderFacts(
                    "canonical.cuda_dense_page@1",
                    device_ordinal=0,
                    max_live_versions_per_lineage=3,
                ),
            ),
            max_admitted_lineages=2,
            state_memory_budget_bytes=4000,
        ),
    )
    match compiled:
        case ESampRequestCudaExecution() as execution:
            pass
        case _:
            raise AssertionError("CUDA specialization selected a non-CUDA execution")
    version_zero = execution.admit(identities, execution_stream)
    assert tuple(int(ref.version_id) for ref in version_zero) == (0, 0)

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
    with torch.cuda.stream(execution_stream):
        source = torch.tensor(
            ((0.2, -0.1, 0.4, 0.3), (-0.5, 0.7, 0.1, -0.2)),
            dtype=torch.float32,
            device=device,
        )
        target = torch.tensor(
            ((0.6, 0.0, -0.3, 0.2), (0.4, -0.2, 0.8, 0.1)),
            dtype=torch.float32,
            device=device,
        )
    hidden_by_resource = {
        hidden_bindings.source: source,
        hidden_bindings.target: target,
    }

    first_rows = PackedPhaseRowBatch(1, 0, 2, row_values)
    first = execution.start_step(
        ESampRequestStepInput(
            first_rows,
            PackedSamplerRowBatch(first_rows, sampler_links),
            CudaPackedHiddenSnapshotBatch.assert_produced_on(
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
                execution_stream,
            ),
        ),
        execution_stream,
    )
    execution.execute_anchor(first, AnchorId(0))
    assert execution.sampler_guidance(
        first,
        sampler_stream,
    ).affected_row_ids == ()
    execution.execute_anchor(first, AnchorId(1))
    execution.complete_sampler(first, sampler_stream)
    execution.execute_anchor(first, AnchorId(2))
    first_result = execution.finish_step(first)
    assert tuple(int(ref.version_id) for ref in first_result.after_refs) == (1, 1)

    second_rows = PackedPhaseRowBatch(1, 1, 2, row_values)
    second = execution.start_step(
        ESampRequestStepInput(
            second_rows,
            PackedSamplerRowBatch(second_rows, sampler_links),
            CudaPackedHiddenSnapshotBatch.assert_produced_on(
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
                execution_stream,
            ),
        ),
        execution_stream,
    )
    execution.execute_anchor(second, AnchorId(0))
    guidance = execution.sampler_guidance(second, sampler_stream)
    assert guidance.affected_row_ids == (0, 1)
    with torch.cuda.stream(sampler_stream):
        guidance.pred_hidden.square().sum()
    execution.execute_anchor(second, AnchorId(1))
    execution.complete_sampler(second, sampler_stream)
    execution.execute_anchor(second, AnchorId(2))
    second_result = execution.finish_step(second)
    assert tuple(int(ref.version_id) for ref in second_result.after_refs) == (2, 2)
    assert all(float(loss) > 0.0 for loss in second_result.losses)

    execution.retire(identities)
    diagnostics = execution.diagnostics()
    assert diagnostics.active_lineages == 0
    assert diagnostics.loss_count == 4
    assert diagnostics.update_count == 4
    assert diagnostics.pending_state_releases == 0
    assert execution.free_state_pages == 6
