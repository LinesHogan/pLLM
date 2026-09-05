"""Evidence 23: multi-target chunk updates, request isolation and retirement."""

import pytest
import torch

from tllm.canonical.state_execution import StateSettlementError


from tllm.canonical.ptts.page_bundle_planning import PageBundleProviderFacts
from tllm.canonical.ptts.page_bundle import PageBundleArena, PagePublication, PageWrite
from tllm.canonical.runtime_facts import CanonicalRuntimeFacts
from tllm.canonical.workloads.inplace_ttt.types import (
    InPlaceTTTCompileOptions, InPlaceTTTConfig,
    InPlaceTTTSpecialization, InPlaceTTTTargetSpecialization,
)
from tllm.canonical.workloads.inplace_ttt.workload import (
    InPlaceTTTWorkload,
)
from tllm.contracts.parameter_overlay import (
    CapturedParameterOperand, ParameterModuleKind, ParameterRole, ParameterTarget,
)
from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.request_state import IndexedRequestIdentityBatch, IndexedRequestIdentityRow
from tllm.contracts.tensor_layout import TensorDType


def test_inplace_ttt_matches_multitarget_oracle_and_cleans_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            (PageBundleProviderFacts("test.cpu.full_root@1", 2),),
            max_admitted_lineages=2, state_memory_budget_bytes=4096,
        ),
    )
    execution = compiled.execution
    assert execution.device.type == "cpu"
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
        step = tuple(
            CapturedParameterOperand(identities[0], target, HiddenPhase.PREFILL,
                hidden[row:row + 1], desired[index, row:row + 1], bases[index])
            for index, target in enumerate(targets)
        )
        before = execution.snapshot(latest)

        def reject_publication(
            self: PageBundleArena, write: PageWrite,
        ) -> PagePublication:
            raise RuntimeError("injected prepublication failure")

        with monkeypatch.context() as fault:
            fault.setattr(PageBundleArena, "publish", reject_publication)
            with pytest.raises(RuntimeError, match="injected prepublication failure"):
                execution.step(step)
        for actual, expected in zip(execution.snapshot(latest), before, strict=True):
            torch.testing.assert_close(actual, expected)
        if row == 0:
            result, = execution.step(step).entries
        else:
            transition, publish = execution.plan.tasks
            rows = IndexedRequestIdentityBatch((IndexedRequestIdentityRow(0, identities[0]),))
            cancelled = execution.begin(rows)
            execution.cancel_step(cancelled)
            with pytest.raises(RuntimeError, match="active"):
                execution.cancel_step(cancelled)
            handle = execution.begin(rows)
            execution.submit_input(handle, step)
            assert handle.cursor.context.data().operands is step
            execution.execute_anchor(handle, transition.anchor_id)
            execution.execute_anchor(handle, publish.anchor_id)
            with pytest.raises(RuntimeError, match="sampler completion"):
                execution.finish_step(handle)
            execution.execute_anchor(handle, execution.sampler_anchor_id)
            execution.complete_sampler(handle)
            publication = handle.cursor.context.publication
            def reject_settlement(self: PageBundleArena, receipt: PagePublication) -> None:
                raise RuntimeError("injected postpublication settlement failure")
            with monkeypatch.context() as fault:
                fault.setattr(PageBundleArena, "settle", reject_settlement)
                with pytest.raises(StateSettlementError) as failed:
                    execution.finish_step(handle)
            assert failed.value.invocation is handle and failed.value.receipt is publication
            release = handle.cursor.context.release
            assert release is not None
            completed = execution.finish_step(handle)
            assert handle.cursor.context.release is release
            (result,) = completed.entries
            with pytest.raises(RuntimeError, match="active"):
                execution.finish_step(handle)
        assert result.before_ref == latest
        latest = result.after_ref
        assert latest.version_id == row + 1
        for index, target_result in enumerate(result.targets):
            torch.testing.assert_close(target_result.output, hidden[row:row + 1] @ bases[index].T)
            assert (target_result.update_count, target_result.trained_rows, target_result.tail_rows) == (
                (0, 0, 1) if row == 0 else (1, 2, 0)
            )
        if row == 0:
            assert not result.weight_changed
        else:
            assert result.weight_changed
            assert publication is not None and publication.after_refs == (latest,)

    # The independent outer-product oracle must be visible only to the trained request.
    deltas = 0.3 * desired.transpose(1, 2) @ hidden
    for request_index, identity in enumerate(identities):
        result, = execution.step(tuple(
            CapturedParameterOperand(identity, target, HiddenPhase.DECODE,
                hidden[:1], desired[index, :1], bases[index])
            for index, target in enumerate(targets)
        )).entries
        assert result.before_ref == result.after_ref == (latest if request_index == 0 else refs[1])
        assert not result.weight_changed
        for index, target_result in enumerate(result.targets):
            weight = bases[index] + deltas[index] if request_index == 0 else bases[index]
            torch.testing.assert_close(target_result.output, hidden[:1] @ weight.T)
            assert target_result.update_count == 0
    assert execution.retire(identities) == (latest, refs[1])
    diagnostics = execution.diagnostics()
    assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
    assert (diagnostics.update_count, diagnostics.trained_rows, diagnostics.fallback_count) == (2, 4, 0)
