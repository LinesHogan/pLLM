"""One real-CUDA PTTS lifecycle proof for canonical dense fast weights."""

from __future__ import annotations

import pytest
import torch

from test.core.oracles.inplace_ttt import oracle_exact_step
from tllm.canonical.parameter_overlay_runtime import (
    CanonicalParameterOverlayRuntimeFacts,
)
from tllm.canonical.ptts import DensePageCudaProviderFacts
from tllm.canonical.workloads.inplace_ttt import (
    CudaInPlaceTTTStepBatchInput,
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
    InPlaceTTTCudaExecution,
    InPlaceTTTOperandPair,
    InPlaceTTTPhase,
    InPlaceTTTSpecialization,
    InPlaceTTTStepInput,
    InPlaceTTTStepBatch,
    InPlaceTTTTargetSpecialization,
    InPlaceTTTWorkload,
)
from tllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
)
from tllm.contracts.parameter_overlay_attachment import (
    ParameterOverlayAttachmentBinding,
    ParameterOverlayTaskBinding,
    ParameterOverlayTaskRole,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.tensor_layout import TensorDType


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_inplace_ttt_keeps_exact_pin_through_version_publication() -> None:
    device = torch.device("cuda", 0)
    stream = torch.cuda.Stream(device=device)
    target = ParameterTarget(
        0,
        ParameterModuleKind.MLP,
        ParameterRole.DOWN_PROJECTION_WEIGHT,
    )
    config = InPlaceTTTConfig(
        (target,),
        chunk_size=2,
        learning_rate=0.3,
        update_decode=True,
    )
    specialization = InPlaceTTTSpecialization(
        (InPlaceTTTTargetSpecialization(target, 3, 2),),
        TensorDType.FLOAT32,
        device,
    )
    workload = InPlaceTTTWorkload(config)
    compiled = workload.compile(
        specialization,
        InPlaceTTTCompileOptions(
            CanonicalParameterOverlayRuntimeFacts(device).timeline(
                (
                    "state/fenced_reclaim@1",
                    "transition/full_root@1",
                )
            ),
            (DensePageCudaProviderFacts("test.cuda.full_root@1", 0, 3),),
            max_admitted_lineages=1,
            state_memory_budget_bytes=4096,
        ),
    )
    match compiled.execution:
        case InPlaceTTTCudaExecution() as execution:
            pass
        case _:
            raise AssertionError("CUDA specialization selected CPU execution")
    bindings = tuple(
        ParameterOverlayAttachmentBinding(
            binding.target,
            binding.operator,
            binding.fields,
        )
        for binding in compiled.state_layout.overlay_bindings
    )
    scheduled = {
        str(task.task_symbol): task
        for task in compiled.plan_bundle.schedule_decision.tasks
    }
    physical = {
        task.task_id: task for task in compiled.plan_bundle.schedule_physical.tasks
    }
    task_bindings = tuple(
        ParameterOverlayTaskBinding(
            role,
            scheduled[symbol].task_id,
            physical[scheduled[symbol].task_id].implementation_id,
            physical[scheduled[symbol].task_id].anchor_id,
        )
        for role, symbol in (
            (ParameterOverlayTaskRole.TRANSITION, "transition"),
            (ParameterOverlayTaskRole.PUBLISH, "publish"),
        )
    )
    anchors = {
        str(anchor.symbol): anchor
        for anchor in compiled.plan_bundle.timeline.anchors
    }
    identity = RequestIdentity("cuda-fast-weight", 0, 0)
    (version_zero,) = execution.admit((identity,), stream)
    base = torch.tensor(
        ((0.2, -0.3, 0.5), (0.4, 0.1, -0.2)),
        dtype=torch.float32,
        device=device,
    )
    first_hidden = torch.tensor(
        ((1.0, -0.5, 0.25),),
        dtype=torch.float32,
        device=device,
    )
    first_target = torch.tensor(
        ((0.5, -0.25),),
        dtype=torch.float32,
        device=device,
    )
    first_oracle = oracle_exact_step(
        first_hidden,
        first_target,
        base,
        chunk_size=2,
        learning_rate=0.3,
    )
    first_handle = execution.begin_batch(
        (identity,),
        stream,
        bindings,
        anchors["request.prepare"].ordinal,
    )
    assert first_handle.refs == (version_zero,)
    torch.testing.assert_close(
        first_handle.deltas[0].tensor,
        torch.zeros_like(base),
    )
    first_input = InPlaceTTTStepInput(
        identity,
        (
            InPlaceTTTOperandPair(
                target,
                InPlaceTTTPhase.PREFILL,
                first_hidden,
                first_target,
                base,
            ),
        ),
    )
    execution.submit_captured(
        first_handle,
        CudaInPlaceTTTStepBatchInput.assert_produced_on(
            InPlaceTTTStepBatch((first_input,)),
            stream,
        ),
        task_bindings[0],
        task_bindings[1],
    )
    with pytest.raises(RuntimeError, match="overlaps"):
        execution.begin_batch(
            (identity,),
            stream,
            bindings,
            anchors["request.prepare"].ordinal,
        )
    execution.complete_sampler(
        first_handle,
        stream,
        anchors["sampler.complete"].ordinal,
    )
    (first,) = execution.finish_step(first_handle).entries
    assert int(first.after_ref.version_id) == 1
    assert first.publication is None

    second_hidden = torch.tensor(
        ((0.2, 0.4, -0.3),),
        dtype=torch.float32,
        device=device,
    )
    second_target = torch.tensor(
        ((0.1, 0.75),),
        dtype=torch.float32,
        device=device,
    )
    second_oracle = oracle_exact_step(
        second_hidden,
        second_target,
        base,
        chunk_size=2,
        learning_rate=0.3,
        past_hidden=first_oracle.tail_hidden,
        past_target=first_oracle.tail_target,
        past_weight=first_oracle.weight,
    )
    second_handle = execution.begin_batch(
        (identity,),
        stream,
        bindings,
        anchors["request.prepare"].ordinal,
    )
    assert second_handle.refs == (first.after_ref,)
    second_input = InPlaceTTTStepInput(
        identity,
        (
            InPlaceTTTOperandPair(
                target,
                InPlaceTTTPhase.DECODE,
                second_hidden,
                second_target,
                base,
            ),
        ),
    )
    execution.submit_captured(
        second_handle,
        CudaInPlaceTTTStepBatchInput.assert_produced_on(
            InPlaceTTTStepBatch((second_input,)),
            stream,
        ),
        task_bindings[0],
        task_bindings[1],
    )
    execution.complete_sampler(
        second_handle,
        stream,
        anchors["sampler.complete"].ordinal,
    )
    (second,) = execution.finish_step(second_handle).entries
    assert int(second.after_ref.version_id) == 2
    assert second.publication is not None
    stream.synchronize()
    torch.testing.assert_close(first.targets[0].output, first_oracle.output)
    torch.testing.assert_close(second.targets[0].output, second_oracle.output)
    snapshot = execution.snapshot(second.after_ref, stream)
    fields = compiled.state_layout.fields_for(target)
    torch.testing.assert_close(
        snapshot[int(fields.delta_weight)],
        second_oracle.weight - base,
    )

    cleanup = execution.retire(
        (identity,),
        anchors["request.finish"].ordinal,
    )
    assert cleanup.retired == (second.after_ref,)
    diagnostics = execution.diagnostics()
    assert diagnostics.active_lineages == 0
    assert diagnostics.update_count == 1
    assert diagnostics.trained_rows == 2
    assert diagnostics.pending_state_releases == 0
    assert diagnostics.fallback_count == 0
