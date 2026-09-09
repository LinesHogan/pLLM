"""Evidence 23: multi-target chunk updates, request isolation and retirement."""


import pytest
import torch
import pllm
from pllm import TensorUpdate
from typing import Annotated
from dataclasses import replace

from pllm.execution.state_execution import StateSettlementError
from pllm.authoring import TensorStepInput, ParameterTargetShape, BackendCompileFacts, StateTensorUpdate
from pllm.contracts.backend_attachment import BackendPlacementProbe, AttachmentTopology, AttachmentGraphMode
from pllm.contracts.state_representation import StateDeviceKind


from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePublication, PageWrite
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand, ParameterModuleKind, ParameterRole, ParameterTarget,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedPhaseRow, PackedSamplerRowBatch
from pllm.contracts.tensor_layout import TensorDType
from test.backend_facts import assert_finite_resources


def _input(operands: tuple[CapturedParameterOperand, ...]) -> TensorStepInput:
    selected = tuple(dict.fromkeys((operand.identity, operand.phase) for operand in operands))
    rows = PackedPhaseRowBatch(1, 0, len(selected), tuple(
        PackedPhaseRow(index, index, 0, phase, identity)
        for index, (identity, phase) in enumerate(selected)))
    return TensorStepInput(rows, PackedSamplerRowBatch(rows, ()), operands=operands)


def test_inplace_ttt_matches_multitarget_oracle_and_cleans_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = tuple(ParameterTarget(
        layer, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT,
    ) for layer in (0, 1))
    config = InPlaceTTTConfig(targets, chunk_size=2, learning_rate=0.3)
    workload = InPlaceTTTWorkload(config)
    facts = BackendCompileFacts(BackendPlacementProbe("test.overlay@1", "a" * 64, "b" * 64, "c" * 64,
        AttachmentTopology.IN_PROCESS, AttachmentGraphMode.EAGER, StateDeviceKind.CPU, 0, 1, 1,
        TensorDType.FLOAT64, 3, 8, (), ()), RuntimeFacts(torch.device("cpu")).timeline(("transition/full_root@1",)),
        (PageBundleProviderFacts("test.cpu.full_root@1", 4),), 2, 4096, 0,
        tuple(ParameterTargetShape(target, 3, 2) for target in targets))
    execution = workload.compile_backend(assert_finite_resources(workload, facts)).execution
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
        if row == 0:
            oversized = tuple(replace(operand, hidden=operand.hidden.expand(9, -1),
                desired=operand.desired.expand(9, -1)) for operand in step)
            with pytest.raises(ValueError, match="packed-row capacity"):
                execution.step(_input(oversized))
            assert execution.arena.latest(latest.lineage_id) == latest

        if row == 0:
            initialize_update = pllm.TensorUpdate.__init__
            calls = []

            def fail_second_target(self, *arguments, **keywords):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("injected second target failure")
                initialize_update(self, *arguments, **keywords)

            free = execution.arena.free_pages
            with monkeypatch.context() as fault:
                fault.setattr(pllm.TensorUpdate, "__init__", fail_second_target)
                with pytest.raises(RuntimeError, match="second target failure"):
                    execution.step(_input(step))
            assert execution.arena.free_pages == free
            assert execution.arena.latest(latest.lineage_id) == latest
            for actual, expected in zip(execution.snapshot(latest), before, strict=True):
                torch.testing.assert_close(actual, expected)

        def reject_publication(
            self: PageBundleArena, write: PageWrite,
        ) -> PagePublication:
            raise RuntimeError("injected prepublication failure")

        with monkeypatch.context() as fault:
            fault.setattr(PageBundleArena, "publish", reject_publication)
            with pytest.raises(RuntimeError, match="injected prepublication failure"):
                execution.step(_input(step))
        for actual, expected in zip(execution.snapshot(latest), before, strict=True):
            torch.testing.assert_close(actual, expected)
        if row == 0:
            result, = execution.step(_input(step)).entries
        else:
            transition, publish = execution.plan.tasks
            value = _input(step)
            rows = value.rows.request_rows
            cancelled = execution.begin(rows)
            execution.cancel_step(cancelled)
            with pytest.raises(RuntimeError, match="active"):
                execution.cancel_step(cancelled)
            handle = execution.begin(rows)
            execution.submit_input(handle, value)
            assert handle.context.selection.rows[0] is value.rows.rows[0]
            assert handle.context.data().step.operands is step
            execution.execute_anchor(handle, transition.anchor_id)
            execution.execute_anchor(handle, publish.anchor_id)
            with pytest.raises(RuntimeError, match="sampler completion"):
                execution.finish_step(handle)
            execution.execute_anchor(handle, execution.sampler_anchor_id)
            execution.complete_sampler(handle)
            publication = handle.context.publication
            def reject_settlement(self: PageBundleArena, receipt: PagePublication) -> None:
                raise RuntimeError("injected postpublication settlement failure")
            with monkeypatch.context() as fault:
                fault.setattr(PageBundleArena, "settle", reject_settlement)
                with pytest.raises(StateSettlementError) as failed:
                    execution.finish_step(handle)
            assert failed.value.invocation is handle and failed.value.receipt is publication
            release = handle.context.release
            assert release is not None
            completed = execution.finish_step(handle)
            assert handle.context.release is release
            (result,) = completed.entries
            with pytest.raises(RuntimeError, match="active"):
                execution.finish_step(handle)
        assert result.before_ref == latest
        latest = result.after_ref
        assert latest.version_id == row + 1
        for index, target in enumerate(result.targets):
            target_result = target.update
            torch.testing.assert_close(target_result.output, hidden[row:row + 1] @ bases[index].T)
            assert (target_result.update_count, target_result.trained_rows, target_result.metrics["tail_rows"]) == (
                (0, 0, 1) if row == 0 else (1, 2, 0)
            )
        if row == 0:
            assert not result.changed
        else:
            assert result.changed
            assert publication is not None and publication.after_refs == (latest,)

    # The independent outer-product oracle must be visible only to the trained request.
    deltas = 0.3 * desired.transpose(1, 2) @ hidden
    for request_index, identity in enumerate(identities):
        result, = execution.step(_input(tuple(
            CapturedParameterOperand(identity, target, HiddenPhase.DECODE,
                hidden[:1], desired[index, :1], bases[index])
            for index, target in enumerate(targets)
        ))).entries
        assert result.before_ref == result.after_ref == (latest if request_index == 0 else refs[1])
        assert not result.changed
        for index, target in enumerate(result.targets):
            target_result = target.update
            weight = bases[index] + deltas[index] if request_index == 0 else bases[index]
            torch.testing.assert_close(target_result.output, hidden[:1] @ weight.T)
            assert target_result.update_count == 0
    assert execution.retire(identities) == (latest, refs[1])
    diagnostics = execution.diagnostics()
    assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
    assert (diagnostics.update_count, diagnostics.trained_rows, diagnostics.fallback_count) == (2, 4, 0)

    # An ordinary fixed-shape state needs neither In-Place-TTT's fields nor its config.
    class LinearOverlay(pllm.Workload):
        weights = pllm.RequestState(shape=(2, 3), dtype=torch.float64)

        @pllm.on(pllm.Phase.MODEL_UPDATE)
        @pllm.overlay(state="weights", targets=(targets[0],))
        def update(self, x: Annotated[torch.Tensor, pllm.Operand.INPUT],
                   y: Annotated[torch.Tensor, pllm.Operand.TARGET],
                   base: Annotated[torch.Tensor, pllm.Operand.WEIGHT],
                   delta: Annotated[torch.Tensor, pllm.Parameter("weights", "value")]) -> TensorUpdate:
            output = x @ (base + delta).T
            delta.add_(0.2 * y.T @ x)
            return TensorUpdate(output, changed=True, update_count=1, trained_rows=x.shape[0])

    ordinary = LinearOverlay().compile_backend(replace(facts, targets=facts.targets[:1])).execution
    requests = (RequestIdentity("first", 0, 0), RequestIdentity("second", 0, 0))
    initial = ordinary.admit(requests)
    result = ordinary.step(_input((CapturedParameterOperand(requests[0], targets[0], HiddenPhase.PREFILL,
        hidden, desired[0], bases[0]),)))
    updated, = result.entries
    torch.testing.assert_close(updated.targets[0].update.output, hidden @ bases[0].T)
    torch.testing.assert_close(ordinary.snapshot(updated.after_ref)[0], 0.2 * desired[0].T @ hidden)
    torch.testing.assert_close(ordinary.snapshot(initial[1])[0], torch.zeros((2, 3), dtype=torch.float64))
    assert ordinary.arena.plan.semantic_graph.states[0].key_fields == (pllm.RequestKeyField.REQUEST_ID,)
    ordinary.retire(requests)

    # A request-scope low-rank overlay shares the same signed binding, pin and isolation rules.
    class LowRankOverlay(pllm.Workload):
        factors = pllm.RequestState({
            "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
            "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
        }, dtype=torch.float64)

        @pllm.on(pllm.Phase.MODEL_UPDATE)
        @pllm.overlay(state="factors", targets=(targets[0],),
                      operator=pllm.ParameterOverlayOperator.LOW_RANK_ADD, factors=("a", "b"), scale=0.5)
        def update(self, x: Annotated[torch.Tensor, pllm.Operand.INPUT],
                   y: Annotated[torch.Tensor, pllm.Operand.TARGET],
                   base: Annotated[torch.Tensor, pllm.Operand.WEIGHT],
                   a: Annotated[torch.Tensor, pllm.Parameter("factors", "a")],
                   b: Annotated[torch.Tensor, pllm.Parameter("factors", "b")]) -> TensorUpdate:
            output = x @ (base + 0.5 * b @ a).T
            a.add_(0.1 * y.T @ x)
            b.add_(0.2 * y.T @ y)
            return TensorUpdate(output, changed=True, update_count=1, trained_rows=x.shape[0])

    _, _, specification = LowRankOverlay().declare(replace(facts, targets=facts.targets[:1]))
    binding, = specification.overlay_bindings
    assert binding.operator is pllm.ParameterOverlayOperator.LOW_RANK_ADD and binding.scale == 0.5
    low_rank = LowRankOverlay().compile_backend(replace(facts, targets=facts.targets[:1])).execution
    initial = low_rank.admit(requests)
    result = low_rank.step(_input((CapturedParameterOperand(requests[0], targets[0], HiddenPhase.PREFILL,
        hidden, desired[0], bases[0]),)))
    updated, = result.entries
    torch.testing.assert_close(updated.targets[0].update.output, hidden @ bases[0].T)
    down, up = low_rank.snapshot(updated.after_ref)
    torch.testing.assert_close(down, 0.1 * desired[0].T @ hidden)
    torch.testing.assert_close(up, 0.2 * desired[0].T @ desired[0])
    assert all(not tensor.any() for tensor in low_rank.snapshot(initial[1]))
    # A phase outside update eligibility applies the exact pinned factors without changing them.
    for request_index, identity in enumerate(requests):
        applied, = low_rank.step(_input((CapturedParameterOperand(identity, targets[0], HiddenPhase.DECODE,
            hidden, desired[0], bases[0]),))).entries
        assert not applied.changed and applied.targets[0].update.update_count == 0
        weight = bases[0] + 0.5 * up @ down if request_index == 0 else bases[0]
        torch.testing.assert_close(applied.targets[0].update.output, hidden @ weight.T)
    low_rank.retire(requests)

    # Externally produced factors (a hypernetwork's output) import through the same signed
    # binding, publication, isolation and retirement rules as optimizer-produced ones.
    # The framework derives the zero workspace of its own canonical import/publish tasks;
    # memory-constrained planning needs no user-declared facts for them.
    class ImportedOverlay(pllm.Workload):
        factors = pllm.RequestState({
            "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
            "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
        }, dtype=torch.float64)

        @property
        def adapter(self) -> pllm.Overlay:
            return pllm.Overlay("factors", (targets[0],),
                                operator=pllm.ParameterOverlayOperator.LOW_RANK_ADD,
                                factors=("a", "b"), scale=0.5)

    generated = (
        torch.tensor([[0.3, -0.1, 0.2], [0.05, 0.4, -0.6]], dtype=torch.float64),
        torch.tensor([[0.7, -0.2], [0.1, 0.5]], dtype=torch.float64),
    )
    imported = ImportedOverlay().compile_backend(assert_finite_resources(
        ImportedOverlay(), replace(facts, targets=facts.targets[:1]))).execution
    prompts = (RequestIdentity("first", 0, 0), RequestIdentity("second", 1, 0))
    initial = imported.admit(prompts)
    rows = PackedPhaseRowBatch(1, 0, 2, tuple(
        PackedPhaseRow(index, index, 0, HiddenPhase.PREFILL, identity)
        for index, identity in enumerate(prompts)))
    keys = tuple(RequestStateKey.project(identity, (RequestKeyField.PROMPT_IDX,)) for identity in prompts)
    # Only the first prompt has externally produced factors; values batch by key.
    update = StateTensorUpdate(tuple(tensor.unsqueeze(0) for tensor in generated), keys[:1])
    publication = imported.step(TensorStepInput(rows, PackedSamplerRowBatch(rows, ()),
        auxiliary=update)).publication
    assert publication is not None and len(publication.after_refs) == 1
    imported_ref, = publication.after_refs
    assert imported_ref.lineage_id == initial[0].lineage_id and imported_ref.version_id == 1
    for actual, expected in zip(imported.snapshot(imported_ref), generated, strict=True):
        torch.testing.assert_close(actual, expected)
    assert imported.arena.latest(initial[1].lineage_id) == initial[1]
    assert all(not tensor.any() for tensor in imported.snapshot(initial[1]))
    # A staged batch does not import twice: the published version no longer matches.
    assert imported.step(TensorStepInput(rows, PackedSamplerRowBatch(rows, ()),
        auxiliary=update)).publication is None
    # The second prompt imports later, on its own, at its own version.
    second = StateTensorUpdate(tuple(tensor.unsqueeze(0) for tensor in generated), keys[1:])
    publication = imported.step(TensorStepInput(rows, PackedSamplerRowBatch(rows, ()),
        auxiliary=second)).publication
    assert publication is not None and len(publication.after_refs) == 1
    second_ref, = publication.after_refs
    assert second_ref.lineage_id == initial[1].lineage_id and second_ref.version_id == 1
    for actual, expected in zip(imported.snapshot(second_ref), generated, strict=True):
        torch.testing.assert_close(actual, expected)
    assert imported.arena.latest(initial[0].lineage_id) == imported_ref
    assert imported.retire(prompts) == (imported_ref, second_ref)
    diagnostics = imported.diagnostics()
    assert diagnostics.update_count == 2
    assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
