"""Evidence 37: CUDA graph replay preserves canonical ESamp training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tllm.canonical.cuda_packed_hidden import CudaPackedHiddenSnapshotBatch
from tllm.canonical.ptts import DensePageCudaProviderFacts
from tllm.canonical.workloads.esamp_request import (
    CanonicalRequestSamplerExecution,
    ESampRequestWorkload,
)
from tllm.canonical.workloads.esamp_request_types import (
    ESampRequestCompileOptions,
    ESampRequestConfig,
    ESampRequestPackedHiddenBindings,
    ESampRequestSpecialization,
    ESampRequestStepInput,
)
from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenPhase, HiddenSite
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


def _timeline() -> SchedulerTimeline:
    return SchedulerTimeline(
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
    )


def _hidden_bindings() -> ESampRequestPackedHiddenBindings:
    return ESampRequestPackedHiddenBindings(
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


def _rows(
    step_id: int,
    identities: tuple[RequestIdentity, RequestIdentity],
) -> PackedPhaseRowBatch:
    return PackedPhaseRowBatch(
        1,
        step_id,
        2,
        (
            PackedPhaseRow(
                0,
                0,
                identities[0].request_id,
                step_id,
                HiddenPhase.DECODE,
                identities[0],
                0,
            ),
            PackedPhaseRow(
                1,
                1,
                identities[1].request_id,
                step_id,
                HiddenPhase.DECODE,
                identities[1],
                1,
            ),
        ),
    )


def _step_input(
    *,
    step_id: int,
    identities: tuple[RequestIdentity, RequestIdentity],
    hidden_bindings: ESampRequestPackedHiddenBindings,
    source: torch.Tensor,
    target: torch.Tensor,
    producer_stream: torch.cuda.Stream,
) -> ESampRequestStepInput:
    rows = _rows(step_id, identities)
    hidden_by_resource = {
        hidden_bindings.source: source,
        hidden_bindings.target: target,
    }
    return ESampRequestStepInput(
        rows,
        PackedSamplerRowBatch(
            rows,
            (
                PackedSamplerRowLink(0, 0),
                PackedSamplerRowLink(1, 1),
            ),
        ),
        CudaPackedHiddenSnapshotBatch.assert_produced_on(
            PackedHiddenSnapshotBatch(
                rows,
                tuple(
                    PackedHiddenSnapshot(
                        resource,
                        hidden_by_resource[resource],
                    )
                    for resource in hidden_bindings.resources
                ),
            ),
            producer_stream,
        ),
    )


def _run_step(
    *,
    execution: CanonicalRequestSamplerExecution,
    step_id: int,
    identities: tuple[RequestIdentity, RequestIdentity],
    hidden_bindings: ESampRequestPackedHiddenBindings,
    source: torch.Tensor,
    target: torch.Tensor,
    execution_stream: torch.cuda.Stream,
    sampler_stream: torch.cuda.Stream,
) -> tuple[float, ...]:
    for last_anchor in (0, 1):
        cancelled = execution.start_step(
            _step_input(
                step_id=step_id, identities=identities, hidden_bindings=hidden_bindings,
                source=source, target=target, producer_stream=execution_stream,
            ),
            execution_stream,
        )
        for anchor in range(last_anchor + 1):
            execution.execute_anchor(cancelled, AnchorId(anchor))
        execution.cancel_step(cancelled)
    handle = execution.start_step(
        _step_input(
            step_id=step_id,
            identities=identities,
            hidden_bindings=hidden_bindings,
            source=source,
            target=target,
            producer_stream=execution_stream,
        ),
        execution_stream,
    )
    execution.execute_anchor(handle, AnchorId(0))
    guidance = execution.sampler_guidance(handle, sampler_stream)
    if guidance.affected_row_ids and guidance.pred_hidden.data_ptr() == 0:
        raise RuntimeError("canonical ESamp guidance did not materialize on CUDA")
    execution.execute_anchor(handle, AnchorId(1))
    execution.complete_sampler(handle, sampler_stream)
    execution.execute_anchor(handle, AnchorId(2))
    result = execution.finish_step(handle)
    execution_stream.synchronize()
    sampler_stream.synchronize()
    return tuple(float(loss.item()) for loss in result.losses)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("canonical ESamp CUDA graph replay smoke requires CUDA")
    device = torch.device("cuda", 0)
    hidden_bindings = _hidden_bindings()
    workload = ESampRequestWorkload(
        ESampRequestConfig(
            rank=2,
            intervention=True,
            intervention_beta=0.2,
            learning_rate=0.05,
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
            timeline=_timeline(),
            providers=(
                DensePageCudaProviderFacts(
                    "canonical.cuda_dense_page@1",
                    device_ordinal=0,
                    max_live_versions_per_lineage=3,
                ),
            ),
            max_admitted_lineages=2,
            state_memory_budget_bytes=10000,
        ),
    )
    execution = compiled.execution
    if execution.device.type != "cuda":
        raise RuntimeError("canonical ESamp did not compile CUDA execution")
    if execution.plan.fallback_count != 0:
        raise RuntimeError("canonical ESamp CUDA plan used fallback tasks")

    execution_stream = torch.cuda.Stream(device=device)
    sampler_stream = torch.cuda.Stream(device=device)
    graph_stream = torch.cuda.Stream(device=device)
    identities = (
        RequestIdentity("cuda-graph-a", 0, 0),
        RequestIdentity("cuda-graph-b", 1, 0),
    )
    with torch.cuda.stream(execution_stream):
        base_source = torch.tensor(
            ((0.2, -0.1, 0.4, 0.3), (-0.5, 0.7, 0.1, -0.2)),
            device=device,
            dtype=torch.float32,
        )
        base_target = torch.tensor(
            ((0.6, 0.0, -0.3, 0.2), (0.4, -0.2, 0.8, 0.1)),
            device=device,
            dtype=torch.float32,
        )
        source = torch.empty_like(base_source)
        target = torch.empty_like(base_target)
    execution.admit(identities, execution_stream)
    execution_stream.synchronize()

    graph_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(graph_stream):
        for _ in range(3):
            source.copy_(base_source)
            target.copy_(base_target)
    torch.cuda.current_stream(device).wait_stream(graph_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=graph_stream):
        source.copy_(base_source)
        target.copy_(base_target)

    hidden_addresses = (source.data_ptr(), target.data_ptr())
    state_addresses = execution.state_pool_addresses
    all_losses: list[float] = []
    for step_id in range(2):
        graph.replay()
        graph_stream.synchronize()
        if (
            hidden_addresses != (source.data_ptr(), target.data_ptr())
            or state_addresses != execution.state_pool_addresses
        ):
            raise RuntimeError("CUDA graph replay changed canonical buffer addresses")
        execution.require_stable_state_pool()
        all_losses.extend(
            _run_step(
                execution=execution,
                step_id=step_id,
                identities=identities,
                hidden_bindings=hidden_bindings,
                source=source,
                target=target,
                execution_stream=execution_stream,
                sampler_stream=sampler_stream,
            )
        )
        execution.collect_completed()

    diagnostics = execution.diagnostics()
    if (
        len(all_losses) != 4
        or not all(loss > 0.0 for loss in all_losses)
        or diagnostics.loss_count != 4
        or diagnostics.update_count != 4
        or diagnostics.completed_sampler_pins != 2
        or diagnostics.active_lineages != 2
        or execution.plan.fallback_count != 0
    ):
        raise RuntimeError("canonical ESamp CUDA graph replay evidence is incomplete")
    cancelled = execution.start_step(
        _step_input(
            step_id=2, identities=identities, hidden_bindings=hidden_bindings,
            source=source, target=target, producer_stream=execution_stream,
        ),
        execution_stream,
    )
    execution.execute_anchor(cancelled, AnchorId(0))
    guidance = execution.sampler_guidance(cancelled, sampler_stream)
    sampler_done = torch.cuda.Event()
    with torch.cuda.stream(sampler_stream):
        torch.cuda._sleep(500_000_000)
        consumed = guidance.pred_hidden.clone()
        sampler_done.record()
    if sampler_done.query():
        raise RuntimeError("cancellation evidence requires an unfinished sampler read")
    execution.retire(identities)
    if not sampler_done.query():
        sampler_stream.synchronize()
        raise RuntimeError("retirement released state before the cancelled sampler completed")
    torch.testing.assert_close(consumed, guidance.pred_hidden)
    execution.collect_completed()
    retired = execution.diagnostics()
    if retired.active_lineages != 0 or retired.pending_state_releases != 0:
        raise RuntimeError("cancelled request retirement left live state or release receipts")
    print(
        {
            "losses": all_losses,
            "loss_count": diagnostics.loss_count,
            "update_count": diagnostics.update_count,
            "fallback_count": execution.plan.fallback_count,
            "hidden_addresses": hidden_addresses,
            "state_addresses": state_addresses,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
