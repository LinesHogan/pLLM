"""Evidence 23: multi-target chunk updates, request isolation and retirement."""

import torch

from tllm.canonical.ptts.dense_page_planning import DensePageCpuProviderFacts
from tllm.canonical.runtime_facts import CanonicalRuntimeFacts
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTCompileOptions, InPlaceTTTConfig, InPlaceTTTOperandPair,
    InPlaceTTTPhase, InPlaceTTTSpecialization, InPlaceTTTStepInput,
    InPlaceTTTTargetSpecialization,
)
from tllm.canonical.workloads.inplace_ttt.workload import (
    CanonicalDenseOverlayExecution, InPlaceTTTWorkload,
)
from tllm.contracts.parameter_overlay import (
    ParameterModuleKind, ParameterRole, ParameterTarget,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.tensor_layout import TensorDType


def test_inplace_ttt_matches_multitarget_oracle_and_cleans_versions() -> None:
    targets = tuple(ParameterTarget(
        layer, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT,
    ) for layer in (0, 1))
    config = InPlaceTTTConfig(targets, chunk_size=2, learning_rate=0.3)
    compiled = InPlaceTTTWorkload(config).compile(
        InPlaceTTTSpecialization(
            tuple(InPlaceTTTTargetSpecialization(target, 3, 2) for target in targets),
            TensorDType.FLOAT64, torch.device("cpu"),
        ),
        InPlaceTTTCompileOptions(
            CanonicalRuntimeFacts(torch.device("cpu")).timeline(("transition/full_root@1",)),
            (DensePageCpuProviderFacts("test.cpu.full_root@1", 2),),
            max_admitted_lineages=2, state_memory_budget_bytes=4096,
        ),
    )
    execution = compiled.execution
    assert isinstance(execution, CanonicalDenseOverlayExecution)
    identities = (RequestIdentity("same", 0, 0), RequestIdentity("same", 0, 1))
    refs = execution.admit(identities)
    bases = torch.tensor([
        [[0.2, -0.3, 0.5], [0.4, 0.1, -0.2]],
        [[-0.1, 0.6, 0.2], [0.3, -0.4, 0.7]],
    ], dtype=torch.float64)
    hidden = torch.tensor([[1.0, -0.5, 0.25], [0.2, 0.4, -0.3]], dtype=torch.float64)
    desired = torch.tensor([
        [[0.5, -0.25], [0.1, 0.75]],
        [[0.2, 0.75], [-0.5, 0.4]],
    ], dtype=torch.float64)
    latest = refs[0]
    for row in range(2):
        result = execution.step(InPlaceTTTStepInput(identities[0], tuple(
            InPlaceTTTOperandPair(target, InPlaceTTTPhase.PREFILL,
                hidden[row:row + 1], desired[index, row:row + 1], bases[index])
            for index, target in enumerate(targets)
        )))
        assert result.before_ref == latest
        latest = result.after_ref
        assert latest.version_id == row + 1
        for index, target_result in enumerate(result.targets):
            torch.testing.assert_close(target_result.output, hidden[row:row + 1] @ bases[index].T)
            assert (target_result.update_count, target_result.trained_rows, target_result.tail_rows) == (
                (0, 0, 1) if row == 0 else (1, 2, 0)
            )
        if row == 0:
            assert result.publication is None
        else:
            assert result.publication is not None and result.publication.ref == latest
            assert tuple(binding.target for binding in result.publication.bindings) == targets

    # The independent outer-product oracle must be visible only to the trained request.
    deltas = 0.3 * desired.transpose(1, 2) @ hidden
    for request_index, identity in enumerate(identities):
        result = execution.step(InPlaceTTTStepInput(identity, tuple(
            InPlaceTTTOperandPair(target, InPlaceTTTPhase.DECODE,
                hidden[:1], desired[index, :1], bases[index])
            for index, target in enumerate(targets)
        )))
        assert result.before_ref == result.after_ref == (latest if request_index == 0 else refs[1])
        assert result.publication is None
        for index, target_result in enumerate(result.targets):
            weight = bases[index] + deltas[index] if request_index == 0 else bases[index]
            torch.testing.assert_close(target_result.output, hidden[:1] @ weight.T)
            assert target_result.update_count == 0
    assert execution.retire(identities).retired == (latest, refs[1])
    diagnostics = execution.diagnostics()
    assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
    assert (diagnostics.update_count, diagnostics.trained_rows, diagnostics.fallback_count) == (2, 4, 0)
