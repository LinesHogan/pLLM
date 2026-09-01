"""One real-CUDA PTTS lifecycle proof for canonical dense fast weights."""

from __future__ import annotations

import pytest
import torch

from test.core.oracles.inplace_ttt import oracle_exact_step
from tllm.canonical.host_runtime import CanonicalHostRuntimeFacts
from tllm.canonical.ptts import DensePageCudaProviderFacts
from tllm.canonical.workloads.inplace_ttt import (
    CudaInPlaceTTTStepInput,
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
    InPlaceTTTCudaExecution,
    InPlaceTTTOperandPair,
    InPlaceTTTPhase,
    InPlaceTTTSpecialization,
    InPlaceTTTStepInput,
    InPlaceTTTTargetSpecialization,
    InPlaceTTTWorkload,
)
from tllm.contracts.parameter_overlay import (
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
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
    config = InPlaceTTTConfig((target,), chunk_size=2, learning_rate=0.3)
    specialization = InPlaceTTTSpecialization(
        (InPlaceTTTTargetSpecialization(target, 3, 2),),
        TensorDType.FLOAT32,
        device,
    )
    workload = InPlaceTTTWorkload(config)
    compiled = workload.compile(
        specialization,
        InPlaceTTTCompileOptions(
            CanonicalHostRuntimeFacts(device).timeline(
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
    first_handle = execution.begin_step(identity, stream)
    assert first_handle.ref == version_zero
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
        CudaInPlaceTTTStepInput.assert_produced_on(first_input, stream),
    )
    with pytest.raises(RuntimeError, match="overlaps"):
        execution.begin_step(identity, stream)
    execution.complete_sampler(first_handle, stream)
    first = execution.finish_step(first_handle)
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
    second_handle = execution.begin_step(identity, stream)
    assert second_handle.ref == first.after_ref
    second_input = InPlaceTTTStepInput(
        identity,
        (
            InPlaceTTTOperandPair(
                target,
                InPlaceTTTPhase.PREFILL,
                second_hidden,
                second_target,
                base,
            ),
        ),
    )
    execution.submit_captured(
        second_handle,
        CudaInPlaceTTTStepInput.assert_produced_on(second_input, stream),
    )
    execution.complete_sampler(second_handle, stream)
    second = execution.finish_step(second_handle)
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

    cleanup = execution.retire((identity,))
    assert cleanup.retired == (second.after_ref,)
    diagnostics = execution.diagnostics()
    assert diagnostics.active_lineages == 0
    assert diagnostics.update_count == 1
    assert diagnostics.trained_rows == 2
    assert diagnostics.pending_state_releases == 0
    assert diagnostics.fallback_count == 0
