"""One Python-Rust/PTTS lifecycle proof for canonical dense fast weights."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from test.core.oracles.inplace_ttt import oracle_exact_step
from tllm.canonical.host_runtime import CanonicalHostRuntimeFacts
from tllm.canonical.ptts.dense_page_planning import DensePageCpuProviderFacts
from tllm.canonical.scheduler import validate_physical_plan
from tllm.canonical.workloads.inplace_ttt import (
    InPlaceTTTApplication,
    InPlaceTTTCompileOptions,
    InPlaceTTTConfig,
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
from tllm.contracts.semantic_graph import AnchorId, StateScope
from tllm.contracts.tensor_layout import TensorDType


def test_inplace_ttt_publishes_one_atomic_multitarget_ptts_version() -> None:
    targets = (
        ParameterTarget(
            0,
            ParameterModuleKind.MLP,
            ParameterRole.DOWN_PROJECTION_WEIGHT,
        ),
        ParameterTarget(
            1,
            ParameterModuleKind.MLP,
            ParameterRole.DOWN_PROJECTION_WEIGHT,
        ),
    )
    config = InPlaceTTTConfig(
        targets,
        chunk_size=2,
        learning_rate=0.3,
        update_prefill=True,
        update_decode=False,
    )
    specialization = InPlaceTTTSpecialization(
        tuple(InPlaceTTTTargetSpecialization(target, 3, 2) for target in targets),
        TensorDType.FLOAT64,
        torch.device("cpu"),
    )
    workload = InPlaceTTTWorkload(config)
    timeline = CanonicalHostRuntimeFacts(torch.device("cpu")).timeline(
        ("transition/full_root@1",)
    )
    compiled = workload.compile(
        specialization,
        InPlaceTTTCompileOptions(
            timeline,
            (DensePageCpuProviderFacts("test.cpu.full_root@1", 2),),
            max_admitted_lineages=2,
            state_memory_budget_bytes=4096,
        ),
    )
    states = {state.resource: state for state in compiled.plan_bundle.semantic_graph.states}
    assert states["dense_fast_weights"].scope is StateScope.REQUEST
    assert compiled.execution.plan.fallback_count == 0
    tampered = replace(
        compiled.plan_bundle.schedule_physical,
        tasks=(
            replace(
                compiled.plan_bundle.schedule_physical.tasks[0],
                anchor_id=AnchorId(
                    int(compiled.plan_bundle.schedule_physical.tasks[0].anchor_id)
                    + 1
                ),
            ),
            *compiled.plan_bundle.schedule_physical.tasks[1:],
        ),
    )
    with pytest.raises(ValueError):
        validate_physical_plan(
            compiled.plan_bundle.semantic_graph,
            compiled.plan_bundle.timeline,
            compiled.plan_bundle.schedule_decision,
            tampered,
        )

    application = InPlaceTTTApplication(workload, compiled)
    first_identity = RequestIdentity("same", 0, 0)
    second_identity = RequestIdentity("same", 0, 1)
    first_ref, second_ref = application.admit((first_identity, second_identity))
    assert int(first_ref.version_id) == 0
    with pytest.raises(RuntimeError, match="entitlement"):
        application.admit((RequestIdentity("overflow", 1, 0),))

    base_weights = (
        torch.tensor(
            ((0.2, -0.3, 0.5), (0.4, 0.1, -0.2)),
            dtype=torch.float64,
        ),
        torch.tensor(
            ((-0.1, 0.6, 0.2), (0.3, -0.4, 0.7)),
            dtype=torch.float64,
        ),
    )
    first_hidden = torch.tensor(((1.0, -0.5, 0.25),), dtype=torch.float64)
    first_desired = (
        torch.tensor(((0.5, -0.25),), dtype=torch.float64),
        torch.tensor(((0.2, 0.75),), dtype=torch.float64),
    )
    first_oracles = tuple(
        oracle_exact_step(
            first_hidden,
            desired,
            base,
            chunk_size=2,
            learning_rate=0.3,
        )
        for desired, base in zip(first_desired, base_weights, strict=True)
    )
    first = application.step(
        InPlaceTTTStepInput(
            first_identity,
            tuple(
                InPlaceTTTOperandPair(
                    target,
                    InPlaceTTTPhase.PREFILL,
                    first_hidden,
                    desired,
                    base,
                )
                for target, desired, base in zip(
                    targets,
                    first_desired,
                    base_weights,
                    strict=True,
                )
            ),
        )
    )
    assert first.before_ref == first_ref
    assert int(first.after_ref.version_id) == 1
    assert first.publication is None
    for result, oracle in zip(first.targets, first_oracles, strict=True):
        torch.testing.assert_close(result.output, oracle.output)
        assert result.update_count == 0
        assert result.tail_rows == 1

    second_hidden = torch.tensor(((0.2, 0.4, -0.3),), dtype=torch.float64)
    second_desired = (
        torch.tensor(((0.1, 0.75),), dtype=torch.float64),
        torch.tensor(((-0.5, 0.4),), dtype=torch.float64),
    )
    second_oracles = tuple(
        oracle_exact_step(
            second_hidden,
            desired,
            base,
            chunk_size=2,
            learning_rate=0.3,
            past_hidden=first_oracle.tail_hidden,
            past_target=first_oracle.tail_target,
            past_weight=first_oracle.weight,
        )
        for desired, base, first_oracle in zip(
            second_desired,
            base_weights,
            first_oracles,
            strict=True,
        )
    )
    second = application.step(
        InPlaceTTTStepInput(
            first_identity,
            tuple(
                InPlaceTTTOperandPair(
                    target,
                    InPlaceTTTPhase.PREFILL,
                    second_hidden,
                    desired,
                    base,
                )
                for target, desired, base in zip(
                    targets,
                    second_desired,
                    base_weights,
                    strict=True,
                )
            ),
        )
    )
    assert second.before_ref == first.after_ref
    assert int(second.after_ref.version_id) == 2
    assert second.publication is not None
    assert second.publication.ref == second.after_ref
    assert tuple(
        binding.target for binding in second.publication.bindings
    ) == targets
    snapshot = compiled.execution.snapshot(second.after_ref)
    for target, result, oracle, base in zip(
        targets,
        second.targets,
        second_oracles,
        base_weights,
        strict=True,
    ):
        torch.testing.assert_close(result.output, oracle.output)
        assert result.update_count == 1
        fields = compiled.state_layout.fields_for(target)
        torch.testing.assert_close(
            snapshot[int(fields.delta_weight)],
            oracle.weight - base,
        )

    decode_hidden = torch.tensor(((0.4, -0.2, 0.1),), dtype=torch.float64)
    decode = application.step(
        InPlaceTTTStepInput(
            second_identity,
            (
                InPlaceTTTOperandPair(
                    targets[0],
                    InPlaceTTTPhase.DECODE,
                    decode_hidden,
                    torch.zeros((1, 2), dtype=torch.float64),
                    base_weights[0],
                ),
            ),
        )
    )
    assert decode.before_ref == second_ref
    assert decode.after_ref == second_ref
    assert decode.publication is None
    torch.testing.assert_close(
        decode.targets[0].output,
        decode_hidden @ base_weights[0].transpose(0, 1),
    )

    cleanup = application.finish((first_identity, second_identity))
    assert cleanup.retired == (second.after_ref, second_ref)
    diagnostics = application.diagnostics()
    assert diagnostics.active_lineages == 0
    assert diagnostics.update_count == 2
    assert diagnostics.trained_rows == 4
    assert diagnostics.fallback_count == 0
