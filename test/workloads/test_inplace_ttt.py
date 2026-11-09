"""In-Place TTT and ordinary overlay workloads against independent outer-product oracles.

Multi-target chunk updates publish one version per request, other requests keep
their entry state, and retirement releases every version.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

import pllm
from pllm import UpdateOutput, testing
from pllm.authoring import StateImport
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.parameter_overlay import (
    Projection,
    ProjectionOperands,
    ProjectionShape,
)
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.overlay_task import ChunkUpdate
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.step_input import StepInput
from pllm.testing import backend_facts
from test.backend_facts import assert_finite_resources
from test.dense_update import DenseUpdate, DenseUpdateConfig


def projection_input(operands: tuple[ProjectionOperands, ...]) -> StepInput:
    """Build step rows for the requests and phases named by projection operands."""
    selected = tuple(dict.fromkeys((operand.identity, operand.phase) for operand in operands))
    identities, phases = zip(*selected, strict=True) if selected else ((), ())
    return testing.step_input(identities, tokens=phases, sampler_order=(), operands=operands)


def test_inplace_ttt_matches_multitarget_oracle_and_cleans_versions() -> None:
    _check_headwise_math_and_tail()
    _check_exact_projection_lifecycle()
    _check_exact_additive_projection()
    targets = tuple(Projection("down_proj", layer) for layer in (0, 1))
    config = DenseUpdateConfig(targets, chunk_size=2, learning_rate=0.3)
    workload = DenseUpdate(config, dtype=torch.float64)
    facts = backend_facts(
        3,
        (),
        model_timeline(torch.device("cpu"), ("transition/full_root@1",)),
        PageBundleProviderFacts("test.cpu.full_root@1", 4),
        memory_bytes=4096,
        max_rows=8,
        dtype=TensorDType.FLOAT64,
        targets=tuple(ProjectionShape(target, 3, 2) for target in targets),
    )
    execution = assert_finite_resources(workload, facts).execution
    visibility = next(
        task.visibility for task in execution.arena.plan.semantic_graph.tasks if task.visibility is not None
    )
    assert visibility.policy == "forward_pinned"
    assert visibility.tail == "transition" and visibility.chunk_size == 0
    pllm.train_in_forward(state="dense_fast_weights", targets=targets, visibility="exact")
    declaration = workload.build(facts).graph
    assert next(task.visibility for task in declaration.tasks if task.visibility is not None) == visibility
    changed = DenseUpdate(replace(config, chunk_size=3)).build(facts).graph
    changed_visibility = next(task.visibility for task in changed.tasks if task.visibility is not None)
    assert changed_visibility.equations_digest != visibility.equations_digest
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
            ProjectionOperands(
                identities[0],
                target,
                HiddenPhase.PREFILL,
                hidden[row : row + 1],
                desired[index, row : row + 1],
                bases[index],
                hidden[row : row + 1].new_full((), False, dtype=torch.bool),
                embedding=torch.empty((0, 0)),
            )
            for index, target in enumerate(targets)
        )
        (result,) = execution.step(projection_input(step)).entries
        assert result.before_ref == latest
        latest = result.after_ref
        assert latest.version_id == row + 1 and result.changed == (row == 1)
        for index, target in enumerate(result.targets):
            torch.testing.assert_close(target.update.output, torch.zeros_like(desired[index, row : row + 1]))
            assert (target.update.update_count, target.update.trained_rows) == ((0, 0) if row == 0 else (1, 2))

    # The independent outer-product oracle must be visible only to the trained request.
    deltas = 0.3 * desired.transpose(1, 2) @ hidden
    for request_index, identity in enumerate(identities):
        value = projection_input(
            tuple(
                ProjectionOperands(
                    identity,
                    target,
                    HiddenPhase.DECODE,
                    hidden[:1],
                    desired[index, :1],
                    bases[index],
                    hidden[:1].new_full((), False, dtype=torch.bool),
                    embedding=torch.empty((0, 0)),
                )
                for index, target in enumerate(targets)
            )
        )
        with patch.object(
            execution.arena, "borrow_roots", side_effect=AssertionError("inactive transition read roots")
        ):
            result = execution.step(replace(value, operands=()))
        assert not result.entries
        ref = latest if request_index == 0 else refs[1]
        assert execution.arena.latest(ref.lineage_id) == ref
        fields = execution.snapshot(ref)
        for index in range(len(targets)):
            weight = deltas[index] if request_index == 0 else torch.zeros_like(bases[index])
            torch.testing.assert_close(fields[4 * index], weight)
    assert execution.retire(identities, wait=True) == (latest, refs[1])
    diagnostics = execution.diagnostics()
    assert diagnostics.active_states == diagnostics.pending_state_releases == 0
    assert (diagnostics.update_count, diagnostics.trained_rows) == (2, 4)

    # A request-scope low-rank overlay shares the same signed binding, pin and isolation rules.
    class LowRankOverlay(pllm.Workload):
        factors = pllm.RequestState(
            {
                "a": pllm.Field(lambda in_features, out_features: (2, in_features), dtype=torch.float64),
                "b": pllm.Field(lambda in_features, out_features: (out_features, 2), dtype=torch.float64),
            },
            dtype=torch.float64,
        )

        @pllm.train_in_forward(
            visibility="forward_pinned",
            state="factors",
            targets=(targets[0],),
            operator="low_rank",
            factors=("a", "b"),
            scale=0.5,
        )
        def update(
            self,
            x: torch.Tensor = pllm.operand("input"),
            y: torch.Tensor = pllm.operand("target"),
            a: torch.Tensor = pllm.parameter("factors", "a"),
            b: torch.Tensor = pllm.parameter("factors", "b"),
        ) -> UpdateOutput:
            output = x @ (b @ a).T
            a.add_(0.1 * y.T @ x)
            b.add_(0.2 * y.T @ y)
            return UpdateOutput(
                output,
                output.new_ones((), dtype=torch.bool),
                output.new_ones((), dtype=torch.int64),
                output.new_full((), x.shape[0], dtype=torch.int64),
            )

    low_rank = (
        LowRankOverlay()
        .compile_backend(replace(facts, probe=replace(facts.probe, targets=facts.probe.targets[:1])))
        .execution
    )
    requests = (RequestIdentity("first", 0, 0), RequestIdentity("second", 0, 0))
    initial = low_rank.admit(requests)
    result = low_rank.step(
        projection_input(
            (
                ProjectionOperands(
                    requests[0],
                    targets[0],
                    HiddenPhase.PREFILL,
                    hidden,
                    desired[0],
                    bases[0],
                    hidden.new_full((), False, dtype=torch.bool),
                    embedding=torch.empty((0, 0)),
                ),
            )
        )
    )
    (updated,) = result.entries
    torch.testing.assert_close(updated.targets[0].update.output, torch.zeros_like(desired[0]))
    down, up = low_rank.snapshot(updated.after_ref)
    torch.testing.assert_close(down, 0.1 * desired[0].T @ hidden)
    torch.testing.assert_close(up, 0.2 * desired[0].T @ desired[0])
    assert all(not tensor.any() for tensor in low_rank.snapshot(initial[1]))
    # A phase outside update eligibility leaves its pinned factors unchanged.
    for request_index, identity in enumerate(requests):
        ref = updated.after_ref if request_index == 0 else initial[1]
        before = low_rank.snapshot(ref)
        value = projection_input(
            (
                ProjectionOperands(
                    identity,
                    targets[0],
                    HiddenPhase.DECODE,
                    hidden,
                    desired[0],
                    bases[0],
                    hidden.new_full((), False, dtype=torch.bool),
                    embedding=torch.empty((0, 0)),
                ),
            )
        )
        result = low_rank.step(replace(value, operands=()))
        assert not result.entries and low_rank.arena.latest(ref.lineage_id) == ref
        torch.testing.assert_close(low_rank.snapshot(ref), before, rtol=0, atol=0)
    low_rank.retire(requests)

    # External factors share the optimizer's publication, isolation and retirement rules.
    # Import validation and publication costs are derived by the framework.
    class ImportedOverlay(pllm.Workload):
        factors = pllm.RequestState(
            {
                "a": pllm.Field(lambda in_features, out_features: (2, in_features), dtype=torch.float64),
                "b": pllm.Field(lambda in_features, out_features: (out_features, 2), dtype=torch.float64),
            },
            init=pllm.Imported(),
            dtype=torch.float64,
        )

        @property
        def adapter(self) -> pllm.model.Adapter:
            return pllm.model.Adapter(
                "factors",
                (targets[0],),
                operator="low_rank",
                factors=("a", "b"),
                scale=0.5,
            )

    generated = (
        torch.tensor([[0.3, -0.1, 0.2], [0.05, 0.4, -0.6]], dtype=torch.float64),
        torch.tensor([[0.7, -0.2], [0.1, 0.5]], dtype=torch.float64),
    )
    imported = assert_finite_resources(
        ImportedOverlay(), replace(facts, probe=replace(facts.probe, targets=facts.probe.targets[:1]))
    ).execution
    prompts = (RequestIdentity("first", 0, 0), RequestIdentity("second", 1, 0))
    initial = imported.admit(prompts)
    # Each prompt imports its externally produced factors on its own, at its own version.
    published = []
    for index, identity in enumerate(prompts):
        update = StateImport(tuple(tensor.unsqueeze(0) for tensor in generated))
        publication = imported.step(
            testing.step_input((identity,), tokens=(HiddenPhase.PREFILL,), sampler_order=(), batch=update)
        ).publication
        (ref,) = publication.after_refs
        assert ref.lineage_id == initial[index].lineage_id and ref.version_id == 1
        for actual, expected in zip(imported.snapshot(ref), generated, strict=True):
            torch.testing.assert_close(actual, expected)
        if index == 0:
            assert imported.arena.latest(initial[1].lineage_id) == initial[1]
            assert all(not tensor.any() for tensor in imported.snapshot(initial[1]))
            # A staged batch does not import twice: the published version no longer matches.
            again = imported.step(
                testing.step_input((identity,), tokens=(HiddenPhase.PREFILL,), sampler_order=(), batch=update)
            )
            assert again.publication is None
        published.append(ref)
    assert imported.retire(prompts, wait=True) == tuple(published)
    diagnostics = imported.diagnostics()
    assert diagnostics.update_count == 2
    assert diagnostics.active_states == diagnostics.pending_state_releases == 0


def _check_headwise_math_and_tail():
    """Head residuals, reconstruction and retained tails match independent FP64 math."""
    from pllm.kernels.head_overlay import head_correction
    from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload

    target = Projection("q_proj", 0)
    workload = GatedDeltaWorkload(GatedDeltaConfig((target,), head_width=3, chunk_size=2, alpha=0.9, beta=0.2))
    generator = torch.Generator().manual_seed(123)
    rows = torch.randn(5, 6, generator=generator)
    weights = torch.randn(2, 3, 3, generator=generator) * 0.05
    tail, length = torch.zeros(1, 6), torch.zeros((), dtype=torch.int64)
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
        expected = torch.cat(outputs)[-len(hidden) :]
        actual = ChunkUpdate(workload.step, 2, ("state", "hidden"))(
            state=weights,
            hidden=hidden,
            _pending_rows_hidden=tail,
            _pending_length=length,
        )
        torch.testing.assert_close(actual.output.double(), expected, rtol=2e-6, atol=5e-7)
        torch.testing.assert_close(weights.double(), state, rtol=2e-6, atol=2e-7)
        torch.testing.assert_close(tail[: len(pending)].double(), pending)
        assert (actual.update_count, actual.trained_rows, int(length)) == (trained // 2, trained, len(pending))
    correction = torch.stack([rows.reshape(-1, 2, 3)[:, h] @ weights[h].T for h in range(2)], dim=1).reshape(-1, 6)
    torch.testing.assert_close(head_correction(rows, weights), correction, rtol=0, atol=0)


def _check_exact_projection_lifecycle():
    """Projection segments isolate requests/targets, retain tails and abort before publication."""
    from pllm.contracts.semantic_graph import AnchorId
    from pllm.execution.overlay_task import ProjectionTask
    from pllm.workloads.gated_delta import GatedDeltaConfig, GatedDeltaWorkload

    targets = tuple(Projection("q_proj", layer) for layer in (0, 2))
    workload = GatedDeltaWorkload(GatedDeltaConfig(targets, head_width=1, chunk_size=2, alpha=1, beta=0.1))
    facts = backend_facts(
        1,
        (),
        model_timeline(torch.device("cpu"), ("transition/full_root@1",)),
        PageBundleProviderFacts("test.cpu.full_root@1", 4),
        memory_bytes=4096,
        max_rows=16,
        targets=tuple(ProjectionShape(target, 1, 1) for target in targets),
    )
    deployment = workload.compile_backend(facts, pllm.ExecutionOptions())
    execution = deployment.execution
    requests = (RequestIdentity("exact-a", 0, 0), RequestIdentity("exact-b", 0, 1))
    refs = execution.admit(requests)
    states = {(identity, target): (0.0, []) for identity in requests for target in targets}
    binding = deployment.attachment.plan.scheduled(deployment.attachment.spec.in_forward)
    # First C=0, then one complete chunk, then a non-power-of-two count with a tail.
    for count in (1, 1, 7):
        operands = tuple(
            ProjectionOperands(
                identity,
                target,
                HiddenPhase.PREFILL,
                torch.full((count, 1), float(i + 1)),
                torch.full((count, 1), float(i + 1)),
                torch.zeros(1, 1),
                torch.full((count, 1), float(i + 1)).new_full((), False, dtype=torch.bool),
                embedding=torch.empty((0, 0)),
            )
            for i, identity in enumerate(requests)
            for target in targets
        )
        value = projection_input(operands)
        invocation = execution.begin(value.rows.request_rows)
        execution.execute_anchor(invocation, AnchorId(0))
        task = ProjectionTask(execution, invocation, value.sampler_rows, binding.task_id, targets)
        _check_projection_reads(task, operands, states, refs, count)
        _publish_projection(execution, invocation)
        result = execution.finish_step(invocation)
        refs = result.publication.after_refs
        for identity, ref in zip(requests, refs, strict=True):
            fields = execution.snapshot(ref)
            for index, target in enumerate(targets):
                state, tail = states[identity, target]
                torch.testing.assert_close(fields[index * 3], torch.tensor([[[state]]]))
                assert fields[index * 3 + 2].item() == len(tail)
    _check_projection_abort(execution, value, targets, binding, refs, operands, requests)
    _check_causality_witness(workload, facts, requests, targets)


def _check_projection_reads(task, operands, states, refs, count):
    """Independent scalar recurrence identifies each exclusive chunk read."""
    invocation = task.context
    targets = tuple(dict.fromkeys(operand.target for operand in operands))
    for target in targets:
        selected = tuple(operand for operand in operands if operand.target == target)
        outputs = task.run(selected)
        for operand in selected:
            state, tail = states[operand.identity, target]
            rows = tail + operand.hidden[:, 0].tolist()
            expected = []
            for start in range(0, len(rows), 2):
                chunk = rows[start : start + 2]
                expected.extend(x * state for x in chunk)
                if len(chunk) == 2:
                    state -= 0.1 * sum(state - x for x in chunk) / 2
            states[operand.identity, target] = state, rows[len(rows) // 2 * 2 :]
            torch.testing.assert_close(outputs[operand.identity, target], torch.tensor(expected[-count:]).view(-1, 1))
        assert invocation.publication is None
        assert all(ref.version_id == before.version_id for ref, before in zip(invocation.pin.refs, refs, strict=True))


def _publish_projection(execution, invocation):
    """Advance the normal completion anchors after all target segments have issued."""
    from pllm.contracts.semantic_graph import AnchorId

    for anchor in execution.arena.plan.timeline.anchors[2:]:
        execution.execute_anchor(invocation, AnchorId(anchor.ordinal))
        if anchor.ordinal == execution.sampler_anchor_id:
            execution.complete_sampler(invocation)


def _check_projection_abort(execution, value, targets, binding, refs, operands, requests):
    """Discard a partial forward without publishing either target's private state."""
    from pllm.contracts.semantic_graph import AnchorId
    from pllm.execution.overlay_task import ProjectionTask

    # A completed projection is still private until the forward's publish anchor.
    before = tuple(execution.snapshot(ref) for ref in refs)
    invocation = execution.begin(value.rows.request_rows)
    execution.execute_anchor(invocation, AnchorId(0))
    task = ProjectionTask(execution, invocation, value.sampler_rows, binding.task_id, targets)
    task.run(tuple(operand for operand in operands if operand.target == targets[0]))
    assert invocation.writing
    execution.cancel_step(invocation)
    for ref, expected in zip(refs, before, strict=True):
        torch.testing.assert_close(execution.snapshot(ref), expected)
    execution.retire(requests)
    assert execution.diagnostics().active_states == 0


def _check_causality_witness(workload, facts, requests, targets):
    """The bound A3 model read observes zero, then the first chunk's update."""
    from pllm.contracts.semantic_graph import AnchorId
    from pllm.execution.overlay_task import ProjectionTask

    witness = replace(workload, config=replace(workload.config, chunk_size=1))
    deployment = witness.compile_backend(facts)
    execution = deployment.execution
    execution.admit(requests[:1])
    operands = tuple(
        ProjectionOperands(
            requests[0],
            target,
            HiddenPhase.PREFILL,
            torch.ones(2, 1),
            torch.ones(2, 1),
            torch.zeros(1, 1),
            torch.ones(2, 1).new_full((), False, dtype=torch.bool),
            embedding=torch.empty((0, 0)),
        )
        for target in targets
    )
    value = projection_input(operands)
    invocation = execution.begin(value.rows.request_rows)
    execution.execute_anchor(invocation, AnchorId(0))
    task = ProjectionTask(
        execution,
        invocation,
        value.sampler_rows,
        deployment.attachment.plan.scheduled(deployment.attachment.spec.in_forward).task_id,
        targets,
    )
    for output in task.run(operands).values():
        torch.testing.assert_close(output[:, 0], torch.tensor([0.0, 0.1]))
    execution.cancel_step(invocation)
    execution.retire(requests[:1])


def _check_exact_additive_projection():
    """Dense Exact uses residual arithmetic with distinct target updates and an unfinished chunk."""
    from pllm.contracts.semantic_graph import AnchorId
    from pllm.execution.overlay_task import ProjectionTask

    targets = tuple(Projection("down_proj", layer) for layer in (0, 1))
    workload = DenseUpdate(DenseUpdateConfig(targets, chunk_size=2, learning_rate=0.01, visibility="exact"))
    facts = backend_facts(
        1,
        (),
        model_timeline(torch.device("cpu"), ("transition/full_root@1",)),
        PageBundleProviderFacts("test.cpu.full_root@1", 4),
        memory_bytes=4096,
        max_rows=8,
        targets=tuple(ProjectionShape(target, 1, 1) for target in targets),
    )
    deployment = workload.compile_backend(facts)
    execution = deployment.execution
    identity = RequestIdentity("additive-exact", 0, 0)
    execution.admit((identity,))
    hidden = torch.arange(1.0, 8).view(-1, 1)
    operands = tuple(
        ProjectionOperands(
            identity,
            target,
            HiddenPhase.PREFILL,
            hidden,
            hidden * (index + 1),
            torch.full((1, 1), 17.0),
            hidden.new_full((), False, dtype=torch.bool),
            embedding=torch.empty((0, 0)),
        )
        for index, target in enumerate(targets)
    )
    value = projection_input(operands)
    invocation = execution.begin(value.rows.request_rows)
    execution.execute_anchor(invocation, AnchorId(0))
    task = ProjectionTask(
        execution,
        invocation,
        value.sampler_rows,
        deployment.attachment.plan.scheduled(deployment.attachment.spec.in_forward).task_id,
        targets,
    )
    expected_states = []
    for operand in operands:
        delta, outputs = torch.zeros(1, 1), []
        for start in range(0, len(hidden), 2):
            x, y = hidden[start : start + 2], operand.desired[start : start + 2]
            outputs.append(x @ delta.T)
            if len(x) == 2:
                delta = delta + 0.01 * y.T @ x
        expected_states.append(delta)
        torch.testing.assert_close(task.run((operand,))[identity, operand.target], torch.cat(outputs))
    _publish_projection(execution, invocation)
    result = execution.finish_step(invocation)
    assert len(result.entries) == 2
    fields = execution.snapshot(result.publication.after_refs[0])
    for index, expected in enumerate(expected_states):
        torch.testing.assert_close(fields[4 * index], expected)
        assert fields[4 * index + 3].item() == 1
    # Exact still needs decode residuals at the projection even when it does not train.
    decode = tuple(
        replace(operand, phase=HiddenPhase.DECODE, hidden=hidden[:1], desired=operand.desired[:1])
        for operand in operands
    )
    value = projection_input(decode)
    invocation = execution.begin(value.rows.request_rows)
    execution.execute_anchor(invocation, AnchorId(0))
    task = ProjectionTask(
        execution,
        invocation,
        value.sampler_rows,
        deployment.attachment.plan.scheduled(deployment.attachment.spec.in_forward).task_id,
        targets,
    )
    for operand, state in zip(decode, expected_states, strict=True):
        torch.testing.assert_close(task.run((operand,))[identity, operand.target], hidden[:1] @ state.T)
    assert not invocation.writing
    _publish_projection(execution, invocation)
    assert execution.finish_step(invocation).publication is None
    execution.retire((identity,))


@pytest.mark.parametrize("span", [1, 4, 7])
def test_prefill_adapters_keep_prompt_chunks_across_scheduler_spans(span):
    from pllm.contracts.semantic_graph import AnchorId
    from pllm.execution.overlay_task import ProjectionTask
    from pllm.workloads.prefill_adapters import TARGETS, PrefillAdaptersWorkload

    workload = PrefillAdaptersWorkload(chunk_size=3, rank=2, dtype=torch.float64)
    facts = backend_facts(
        3,
        (),
        model_timeline(torch.device("cpu"), ("transition/full_root@1",)),
        PageBundleProviderFacts("test.cpu.full_root@1", 4),
        memory_bytes=1 << 20,
        max_rows=32,
        dtype=TensorDType.FLOAT64,
        layer_count=10,
        targets=tuple(ProjectionShape(target, 3, 2) for target in TARGETS),
    )
    deployed = assert_finite_resources(workload, facts)
    execution = deployed.execution
    ids = (RequestIdentity("prompt", 0, 0), RequestIdentity("decode", 1, 0))
    refs = execution.admit(ids)
    initial, frozen = (execution.snapshot(ref) for ref in refs)
    generator = torch.Generator().manual_seed(19)
    x, desired, base = (
        torch.randn(shape, generator=generator, dtype=torch.float64) for shape in ((11, 3), (11, 2), (2, 3))
    )
    expected, factors = [], []
    for target_index in range(len(TARGETS)):
        a, b = (v.clone().requires_grad_() for v in initial[target_index * 5 : target_index * 5 + 2])
        output = []
        for start in range(0, len(x), workload.chunk_size):
            rows, wanted = x[start : start + 3], desired[start : start + 3]
            delta = rows @ a.T @ b.T
            output.append(delta.detach())
            loss = (rows @ base.T + delta - wanted).square().mean()
            gradients = torch.autograd.grad(loss, (a, b))
            with torch.no_grad():
                a.sub_(workload.learning_rate * gradients[0])
                b.sub_(workload.learning_rate * gradients[1])
        expected.append(torch.cat(output))
        factors.append((a.detach(), b.detach()))
    for start in range(0, len(x), span):
        end = min(start + span, len(x))
        operands = tuple(
            ProjectionOperands(
                identity,
                target,
                phase,
                x[start:end],
                desired[start:end],
                base,
                x[start:end].new_full((), end == len(x), dtype=torch.bool),
                embedding=torch.empty((0, 0)),
            )
            for identity, phase in zip(ids, (HiddenPhase.PREFILL, HiddenPhase.DECODE), strict=True)
            for target in TARGETS
        )
        value = projection_input(operands)
        invocation = execution.begin(value.rows.request_rows)
        execution.execute_anchor(invocation, AnchorId(0))
        task = ProjectionTask(
            execution,
            invocation,
            value.sampler_rows,
            deployed.attachment.plan.scheduled(deployed.attachment.spec.in_forward).task_id,
            TARGETS,
        )
        outputs = task.run(operands)
        for index, target in enumerate(TARGETS):
            torch.testing.assert_close(outputs[ids[0], target], expected[index][start:end], rtol=1e-12, atol=1e-12)
        _publish_projection(execution, invocation)
        execution.finish_step(invocation)
        fields = execution.snapshot(execution.arena.latest(refs[0].lineage_id))
        for index in range(len(TARGETS)):
            tail = 0 if end == len(x) else end % workload.chunk_size
            assert fields[index * 5 + 4].item() == tail
    for index, pair in enumerate(factors):
        torch.testing.assert_close(fields[index * 5 : index * 5 + 2], pair, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(execution.snapshot(execution.arena.latest(refs[1].lineage_id)), frozen)
    assert execution.diagnostics().update_count == 4 * len(TARGETS)
    assert execution.diagnostics().trained_rows == len(x) * len(TARGETS)
    execution.retire(ids, wait=True)
    assert execution.diagnostics().active_states == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph operand lifetime")
@pytest.mark.gpu
def test_prompt_end_operand_stays_live_during_replay():
    from pllm.workloads.prefill_adapters import TARGETS, PrefillAdaptersWorkload

    workload = PrefillAdaptersWorkload(chunk_size=3, rank=2, dtype=torch.float32)
    math = workload.build(
        backend_facts(
            3,
            (),
            model_timeline(torch.device("cpu"), ("transition/full_root@1",)),
            None,
            targets=tuple(ProjectionShape(target, 3, 2) for target in TARGETS),
        )
    ).overlay_math
    x, base = torch.ones(1, 3, device="cuda"), torch.zeros(2, 3, device="cuda")
    values = dict(
        a=torch.ones(2, 3, device="cuda"),
        b=torch.zeros(2, 2, device="cuda"),
        _pending_grad_a=torch.zeros(2, 3, device="cuda"),
        _pending_grad_b=torch.zeros(2, 2, device="cuda"),
        _pending_length=torch.zeros((), dtype=torch.int64, device="cuda"),
    )
    operand = ProjectionOperands(
        RequestIdentity("graph", 0, 0),
        TARGETS[0],
        HiddenPhase.PREFILL,
        x,
        x[:, :2],
        base,
        torch.zeros((), dtype=torch.bool, device="cuda"),
        embedding=x.new_empty((0, 0)),
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        math.result(operand, values)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), torch.no_grad():
        result = math.result(operand, values)
    for name, value in values.items():
        value.fill_(1 if name == "a" else 0)
    graph.replay()
    assert values["_pending_length"].item() == 1 and result.update_count.item() == 0
    operand.prompt_end.fill_(True)
    graph.replay()
    assert values["_pending_length"].item() == 0 and result.update_count.item() == 1
    assert result.trained_rows.item() == 2
    torch.testing.assert_close(values["b"], torch.full_like(values["b"], 0.003))


class PaperLinear(torch.nn.Linear):
    """Return (output, bias) like vLLM's linear layers."""

    def forward(self, input):
        return super().forward(input), None


class PaperMlp(torch.nn.Module):
    """Small gated MLP; the fake decoder uses the same projection interface as vLLM."""

    def __init__(self):
        super().__init__()
        self.gate, self.up = torch.nn.Linear(3, 5, bias=False), torch.nn.Linear(3, 5, bias=False)
        self.down_proj = PaperLinear(5, 3, bias=False)
        self.down_proj.params_dtype = torch.float32
        self.down_proj.input_size_per_partition = 5

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate(x)) * self.up(x))[0]


class PaperDecoder(torch.nn.Module):
    """Pointwise two-layer CPU decoder whose embedding stream differs from every MLP input."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.ModuleDict({"input_layernorm": torch.nn.LayerNorm(3), "mlp": PaperMlp()}) for _ in range(2)
        )

    def forward(self, x):
        for layer in self.layers:
            x = x + layer["mlp"](layer["input_layernorm"](x))
        return x

    def projection(self, target):
        mlp = self.layers[target.layer]["mlp"]
        return mlp.down_proj, 0, 3, mlp


def paper_oracle(body, x0, convs, projs, track, eta):
    """Paper equations with an explicit kernel sum and autograd of the negative inner product."""
    x, states, local_norms = x0, [], []
    for layer, conv, proj in zip(body.layers, convs, projs, strict=True):
        h = layer["input_layernorm"](x)
        mlp = layer["mlp"]
        z = torch.nn.functional.silu(mlp.gate(h)) * mlp.up(h)
        source = x0 if track == "T2" else h
        delta, outputs = torch.zeros(3, 5), []
        prefixes = [delta]
        for start in range(0, len(x), 4):
            end = min(start + 4, len(x))
            outputs.append(z[start:end] @ (mlp.down_proj.weight.T + delta.T))
            if end - start == 4:
                padded = torch.nn.functional.pad(source[start:end], (0, 0, 2, 2))
                v = sum(padded[k : k + 4] * conv[:, 0, k] for k in range(5)) @ proj
                weight = torch.zeros(3, 5, requires_grad=True)
                (gradient,) = torch.autograd.grad(-((z[start:end] @ weight.T) * v).sum(), (weight,))
                update = -gradient
                norm = float(update.norm())
                local_norms.append(norm)
                if track == "T2" and norm > 1e-5:
                    update *= 1e-5 / norm
                delta = delta + eta * update
                prefixes.append(delta)
        x = x + torch.cat(outputs)
        states.append(prefixes)
    return x, states, local_norms


def run_paper_schedule(deployed, body, overlay, ids, prompts, expected, track):
    """Check every completed chunk's state and each token's exclusive read across mixed scheduling."""
    from pllm.execution.overlay_task import ProjectionTask
    from pllm.testing_contracts import scheduled_step

    execution = deployed.execution
    refs = tuple(ref for ref in execution.admit(ids) if execution.arena.root(ref).resource == "dense_fast_weights")
    task_id = deployed.attachment.plan.scheduled(deployed.attachment.spec.in_forward).task_id
    targets = tuple(binding.target for binding in deployed.attachment.spec.overlay_bindings)
    done = [0, 0]
    output_error = state_error = 0.0
    # Request b starts with a longer prompt; request a's first chunk completes in decode.
    for counts in ((3, 5), (2, 1), (1, 1), (1, 1), (1, 1), (1, 0)):
        value = scheduled_step(ids, list(counts), [3, 5], done, {})
        context = execution.start_step(value)
        execution.advance_to(context, "request.prepare")
        task = ProjectionTask(execution, context, value.sampler_rows, task_id, targets)
        overlay.prepare(value.rows, context, execution.directory, task)
        result = body(torch.cat(tuple(x[start : start + n] for x, start, n in zip(prompts, done, counts, strict=True))))
        overlay.take_operands()
        overlay.clear()
        _publish_projection(execution, context)
        execution.finish_step(context)
        for i, (ref, actual) in enumerate(zip(refs, torch.split(result, counts), strict=True)):
            end = done[i] + counts[i]
            wanted = expected[i][0][done[i] : end]
            torch.testing.assert_close(actual, wanted, rtol=2e-6, atol=2e-7)
            if actual.numel():
                output_error = max(output_error, (actual - wanted).abs().max().item())
            fields = execution.snapshot(execution.arena.latest(ref.lineage_id))
            for layer in range(2):
                state = expected[i][1][layer][end // 4]
                torch.testing.assert_close(fields[4 * layer], state, rtol=1e-5, atol=1e-6 if track == "T1" else 2e-11)
                state_error = max(state_error, (fields[4 * layer] - state).abs().max().item())
                assert fields[4 * layer + 3] == end % 4
            done[i] = end
    execution.retire(ids, wait=True)
    print(f"{track}: output max_abs={output_error:.9g}, state max_abs={state_error:.9g}")


@pytest.mark.parametrize("track,scale", (("T1", 1.0), ("T2", 1.0), ("T2", 1e-8), ("T2", 0.0)))
def test_paper_ttt_matches_fake_decoder_oracle(tmp_path, track, scale):
    """A paper-formula oracle checks X0 routing, local clipping, tails, visibility and fresh requests."""
    from types import SimpleNamespace

    from safetensors.torch import save_file

    from pllm.backends.vllm.parameter_overlay import ParameterOverlay
    from pllm.workloads import InPlaceTTTConfig, InPlaceTTTWorkload

    torch.manual_seed(19)
    body = PaperDecoder()
    convs = tuple(torch.randn(3, 1, 5) * scale for _ in range(2))
    projs = tuple(torch.randn(3, 3) for _ in range(2))
    save_file(
        {
            f"model.layers.{i}.mlp.ttt_{name}.weight": tensor
            for i, pair in enumerate(zip(convs, projs, strict=True))
            for name, tensor in zip(("conv", "proj"), pair, strict=True)
        },
        tmp_path / "model.safetensors",
    )
    targets = tuple(Projection("down_proj", i) for i in range(2))
    workload = InPlaceTTTWorkload(InPlaceTTTConfig(targets, 4, 3.0, track=track))
    facts = replace(
        backend_facts(
            3,
            timeline=model_timeline(torch.device("cpu"), ("transition/full_root@1",)),
            memory_bytes=1 << 20,
            max_rows=16,
            layer_count=2,
            targets=tuple(ProjectionShape(t, 5, 3) for t in targets),
        ),
        checkpoint=str(tmp_path),
    )
    deployed = workload.compile_backend(facts)
    execution = deployed.execution
    execution.admit_instances()
    ids = (RequestIdentity("a", 0, 0), RequestIdentity("b", 1, 0))
    prompts = (torch.randn(9, 3), torch.randn(9, 3))
    expected = tuple(paper_oracle(body, x, convs, projs, track, 3.0) for x in prompts)
    decoder = SimpleNamespace(model=SimpleNamespace(model=body), projection=body.projection)
    bindings = deployed.attachment.spec.overlay_bindings
    overlay = ParameterOverlay(decoder, bindings, execution.arena, 16, 2, 0, (), in_forward=True, graphs=False)
    try:
        # A partial prefill chunk completes in decode; the completing token reads old state.
        with (
            patch("pllm.backends.vllm.parameter_overlay.frozen_weight", side_effect=lambda m: m.weight.detach()),
            torch.no_grad(),
        ):
            for _ in range(2):
                run_paper_schedule(deployed, body, overlay, ids, prompts, expected, track)
    finally:
        overlay.close()
    if track == "T2":
        assert all((norm > 1e-5) == (scale == 1.0) for _, _, norms in expected for norm in norms)
    assert not body.layers[0]["input_layernorm"]._forward_pre_hooks
