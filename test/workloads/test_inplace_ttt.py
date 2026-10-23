"""In-Place TTT and ordinary overlay workloads against independent outer-product oracles.

Multi-target chunk updates publish one version per request, other requests keep
their exact base weights, and retirement releases every version.
"""

from dataclasses import replace
from typing import Annotated

import pytest
import torch

import pllm
from pllm import TensorUpdate
from pllm.authoring import ParameterTargetShape, StateTensorUpdate, TensorStepInput
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
)
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.request_state import RequestStateKey
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.visibility import explain_visibility
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload
from test.backend_facts import assert_finite_resources, backend_facts


class _ShiftedUpdate(InPlaceTTTWorkload):
    """Change only a helper's mathematics to check the bound source identity."""

    def step(
        self, delta: torch.Tensor, inputs: torch.Tensor, targets: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use a different increment without changing the transition method."""
        return delta + self.config.learning_rate * torch.mm(targets.T, inputs) + 1, torch.mm(inputs, (base + delta).T)


def _input(operands: tuple[CapturedParameterOperand, ...]) -> TensorStepInput:
    selected = tuple(dict.fromkeys((operand.identity, operand.phase) for operand in operands))
    rows = PackedPhaseRowBatch(
        1,
        0,
        len(selected),
        tuple(PackedPhaseRow(index, index, 0, phase, identity) for index, (identity, phase) in enumerate(selected)),
    )
    return TensorStepInput(rows, PackedSamplerRowBatch(rows, ()), operands=operands)


def test_inplace_ttt_matches_multitarget_oracle_and_cleans_versions() -> None:
    _check_headwise_math_and_tail()
    targets = tuple(
        ParameterTarget(layer, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT) for layer in (0, 1)
    )
    config = InPlaceTTTConfig(targets, chunk_size=2, learning_rate=0.3)
    workload = InPlaceTTTWorkload(config)
    facts = backend_facts(
        3,
        (),
        RuntimeFacts(torch.device("cpu")).timeline(("transition/full_root@1",)),
        PageBundleProviderFacts("test.cpu.full_root@1", 4),
        memory_bytes=4096,
        max_rows=8,
        dtype=TensorDType.FLOAT64,
        targets=tuple(ParameterTargetShape(target, 3, 2) for target in targets),
    )
    execution = workload.compile_backend(assert_finite_resources(workload, facts)).execution
    visibility = next(
        task.visibility for task in execution.arena.plan.semantic_graph.tasks if task.visibility is not None
    )
    assert visibility.policy == pllm.Visibility.forward_pinned()
    assert visibility.tail == "transition" and visibility.chunk_size == 0
    assert "model[" in explain_visibility(visibility) and "update[" in explain_visibility(visibility)
    with pytest.raises(ValueError, match="use Visibility.forward_pinned"):
        pllm.overlay(state="dense_fast_weights", targets=targets, visibility=pllm.Exact())
    declaration = workload.declaration().graph
    assert next(task.visibility for task in declaration.tasks if task.visibility is not None) == visibility
    changed = InPlaceTTTWorkload(replace(config, chunk_size=3)).declaration().graph
    changed_visibility = next(task.visibility for task in changed.tasks if task.visibility is not None)
    assert changed_visibility.equations_digest != visibility.equations_digest
    helper = _ShiftedUpdate(config).declaration().graph
    helper_visibility = next(task.visibility for task in helper.tasks if task.visibility is not None)
    assert helper_visibility.equations_digest != visibility.equations_digest
    identities = (RequestIdentity("same", 0, 0), RequestIdentity("same", 0, 1))
    refs = execution.admit(identities)
    bases = torch.tensor(
        [[[0.2, -0.3, 0.5], [0.4, 0.1, -0.2]], [[-0.1, 0.6, 0.2], [0.3, -0.4, 0.7]]], dtype=torch.float64
    )
    hidden = torch.tensor([[1.0, -0.5, 0.25], [0.2, 0.4, -0.3]], dtype=torch.float64)
    desired = torch.tensor([[[0.5, -0.25], [0.1, 0.75]], [[0.2, 0.75], [-0.5, 0.4]]], dtype=torch.float64)
    latest = refs[0]
    # Prefill rows fill a chunk of two; the update publishes once the chunk is complete.
    for row in range(2):
        step = tuple(
            CapturedParameterOperand(
                identities[0],
                target,
                HiddenPhase.PREFILL,
                hidden[row : row + 1],
                desired[index, row : row + 1],
                bases[index],
            )
            for index, target in enumerate(targets)
        )
        if row == 0:
            oversized = tuple(
                replace(operand, hidden=operand.hidden.expand(9, -1), desired=operand.desired.expand(9, -1))
                for operand in step
            )
            with pytest.raises(ValueError):  # more rows than the signed packed-row capacity
                execution.step(_input(oversized))
            assert execution.arena.latest(latest.lineage_id) == latest
        (result,) = execution.step(_input(step)).entries
        assert result.before_ref == latest
        latest = result.after_ref
        assert latest.version_id == row + 1 and result.changed == (row == 1)
        for index, target in enumerate(result.targets):
            torch.testing.assert_close(target.update.output, hidden[row : row + 1] @ bases[index].T)
            assert (target.update.update_count, target.update.trained_rows) == ((0, 0) if row == 0 else (1, 2))

    # The independent outer-product oracle must be visible only to the trained request.
    deltas = 0.3 * desired.transpose(1, 2) @ hidden
    for request_index, identity in enumerate(identities):
        (result,) = execution.step(
            _input(
                tuple(
                    CapturedParameterOperand(
                        identity, target, HiddenPhase.DECODE, hidden[:1], desired[index, :1], bases[index]
                    )
                    for index, target in enumerate(targets)
                )
            )
        ).entries
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

    # A request-scope low-rank overlay shares the same signed binding, pin and isolation rules.
    class LowRankOverlay(pllm.Workload):
        factors = pllm.RequestState(
            {
                "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
                "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
            },
            dtype=torch.float64,
        )

        @pllm.on(pllm.Phase.MODEL_UPDATE)
        @pllm.overlay(
            visibility=pllm.Visibility.forward_pinned(),
            state="factors",
            targets=(targets[0],),
            operator=pllm.ParameterOverlayOperator.LOW_RANK_ADD,
            factors=("a", "b"),
            scale=0.5,
        )
        def update(
            self,
            x: Annotated[torch.Tensor, pllm.Operand.INPUT],
            y: Annotated[torch.Tensor, pllm.Operand.TARGET],
            base: Annotated[torch.Tensor, pllm.Operand.WEIGHT],
            a: Annotated[torch.Tensor, pllm.Parameter("factors", "a")],
            b: Annotated[torch.Tensor, pllm.Parameter("factors", "b")],
        ) -> TensorUpdate:
            output = x @ (base + 0.5 * b @ a).T
            a.add_(0.1 * y.T @ x)
            b.add_(0.2 * y.T @ y)
            return TensorUpdate(output, changed=True, update_count=1, trained_rows=x.shape[0])

    low_rank = LowRankOverlay().compile_backend(replace(facts, targets=facts.targets[:1])).execution
    requests = (RequestIdentity("first", 0, 0), RequestIdentity("second", 0, 0))
    initial = low_rank.admit(requests)
    result = low_rank.step(
        _input((CapturedParameterOperand(requests[0], targets[0], HiddenPhase.PREFILL, hidden, desired[0], bases[0]),))
    )
    (updated,) = result.entries
    torch.testing.assert_close(updated.targets[0].update.output, hidden @ bases[0].T)
    down, up = low_rank.snapshot(updated.after_ref)
    torch.testing.assert_close(down, 0.1 * desired[0].T @ hidden)
    torch.testing.assert_close(up, 0.2 * desired[0].T @ desired[0])
    assert all(not tensor.any() for tensor in low_rank.snapshot(initial[1]))
    # A phase outside update eligibility applies the exact pinned factors without changing them.
    for request_index, identity in enumerate(requests):
        (applied,) = low_rank.step(
            _input((CapturedParameterOperand(identity, targets[0], HiddenPhase.DECODE, hidden, desired[0], bases[0]),))
        ).entries
        assert not applied.changed and applied.targets[0].update.update_count == 0
        weight = bases[0] + 0.5 * up @ down if request_index == 0 else bases[0]
        torch.testing.assert_close(applied.targets[0].update.output, hidden @ weight.T)
    low_rank.retire(requests)

    # External factors share the optimizer's publication, isolation and retirement rules.
    # Import validation and publication costs are derived by the framework.
    class ImportedOverlay(pllm.Workload):
        factors = pllm.RequestState(
            {
                "a": pllm.Buffer(lambda inputs, outputs: (2, inputs)),
                "b": pllm.Buffer(lambda inputs, outputs: (outputs, 2)),
            },
            dtype=torch.float64,
        )

        @property
        def adapter(self) -> pllm.Overlay:
            return pllm.Overlay(
                "factors",
                (targets[0],),
                operator=pllm.ParameterOverlayOperator.LOW_RANK_ADD,
                factors=("a", "b"),
                scale=0.5,
            )

    generated = (
        torch.tensor([[0.3, -0.1, 0.2], [0.05, 0.4, -0.6]], dtype=torch.float64),
        torch.tensor([[0.7, -0.2], [0.1, 0.5]], dtype=torch.float64),
    )
    imported = (
        ImportedOverlay()
        .compile_backend(assert_finite_resources(ImportedOverlay(), replace(facts, targets=facts.targets[:1])))
        .execution
    )
    prompts = (RequestIdentity("first", 0, 0), RequestIdentity("second", 1, 0))
    initial = imported.admit(prompts)
    rows = PackedPhaseRowBatch(
        1,
        0,
        2,
        tuple(PackedPhaseRow(index, index, 0, HiddenPhase.PREFILL, identity) for index, identity in enumerate(prompts)),
    )
    keys = tuple(RequestStateKey.project(identity, (RequestKeyField.PROMPT_IDX,)) for identity in prompts)
    # Each prompt imports its externally produced factors on its own, at its own version.
    published = []
    for index, key in enumerate(keys):
        update = StateTensorUpdate(tuple(tensor.unsqueeze(0) for tensor in generated), (key,))
        publication = imported.step(
            TensorStepInput(rows, PackedSamplerRowBatch(rows, ()), auxiliary=update)
        ).publication
        (ref,) = publication.after_refs
        assert ref.lineage_id == initial[index].lineage_id and ref.version_id == 1
        for actual, expected in zip(imported.snapshot(ref), generated, strict=True):
            torch.testing.assert_close(actual, expected)
        if index == 0:
            assert imported.arena.latest(initial[1].lineage_id) == initial[1]
            assert all(not tensor.any() for tensor in imported.snapshot(initial[1]))
            # A staged batch does not import twice: the published version no longer matches.
            again = imported.step(TensorStepInput(rows, PackedSamplerRowBatch(rows, ()), auxiliary=update))
            assert again.publication is None
        published.append(ref)
    assert imported.retire(prompts) == tuple(published)
    diagnostics = imported.diagnostics()
    assert diagnostics.update_count == 2
    assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0


def _check_headwise_math_and_tail():
    """Head residuals, reconstruction and retained tails match independent FP64 math."""
    from pllm.kernels.head_overlay import head_correction
    from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload

    target = ParameterTarget(0, ParameterModuleKind.ATTENTION, ParameterRole.QUERY_PROJECTION_WEIGHT)
    workload = GatedDeltaWorkload(GatedDeltaConfig((target,), head_width=3, chunk_size=2, alpha=0.9, beta=0.2))
    generator = torch.Generator().manual_seed(123)
    base = torch.randn(6, 6, generator=generator)
    rows = torch.randn(5, 6, generator=generator)
    weights, tail, length = torch.zeros(2, 3, 3), torch.zeros(1, 6), torch.zeros((), dtype=torch.int64)
    state, pending = weights.double(), rows[:0].double()
    for hidden in (rows[:1], rows[1:]):
        joined = torch.cat((pending, hidden.double()))
        trained = len(joined) // 2 * 2
        outputs = []
        for start in range(0, len(joined), 2):
            x = joined[start : start + 2].reshape(-1, 2, 3)
            outputs.append(torch.stack([x[:, h] @ state[h].T for h in range(2)], dim=1).reshape(-1, 6))
            if start < trained:
                decayed = 0.9 * state
                gradient = torch.zeros_like(state)
                for row in x:
                    for head in range(2):
                        key = row[head] / row[head].norm().clamp_min(1e-12)
                        gradient[head] += torch.outer(decayed[head] @ key - row[head], key)
                state = decayed - 0.2 / 2 * gradient
        pending = joined[trained:]
        expected = hidden.double() @ base.double().T + torch.cat(outputs)[-len(hidden) :]
        actual = workload.transition(hidden, base, weights, tail, length)
        torch.testing.assert_close(actual.output.double(), expected, rtol=2e-6, atol=5e-7)
        torch.testing.assert_close(weights.double(), state, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(tail[: len(pending)].double(), pending)
        assert (actual.update_count, actual.trained_rows, int(length)) == (trained // 2, trained, len(pending))
    correction = torch.stack([rows.reshape(-1, 2, 3)[:, h] @ weights[h].T for h in range(2)], dim=1).reshape(-1, 6)
    torch.testing.assert_close(head_correction(rows, weights), correction, rtol=0, atol=0)
