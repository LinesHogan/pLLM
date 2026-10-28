"""Routed overlay reads and explicit boundaries reuse one private root and one publication."""

from dataclasses import dataclass, replace
from unittest.mock import patch

import pytest
import torch

import pllm
from pllm.contracts.codec import decode_boundary, encode_boundary
from pllm.contracts.parameter_overlay import (
    CapturedParameterOperand,
    ParameterModuleKind,
    ParameterRole,
    ParameterTarget,
    ParameterTargetShape,
)
from pllm.contracts.semantic_graph import AnchorId
from pllm.contracts.state_representation import StateDeviceKind
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from test.backend_facts import assert_finite_resources
from test.workloads.test_recirculation import FACTS, IDENTITIES, INPUTS, ROWS, VALUE, Decoder

TARGET = ParameterTarget(1, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
CAPABILITIES = tuple(sorted(("model/tokenwise_replay@1", "transition/full_root@1", "state/fenced_reclaim@1")))
ROUTE_FACTS = replace(
    FACTS,
    probe=replace(FACTS.probe, capabilities=CAPABILITIES, targets=(ParameterTargetShape(TARGET, 4, 4),)),
    timeline=model_timeline(torch.device("cpu"), CAPABILITIES),
    providers=(PageBundleProviderFacts("test.route@1", 16, StateDeviceKind.CPU),),
    state_memory_budget_bytes=1 << 20,
)


@dataclass(frozen=True)
class Updating(pllm.Workload):
    visibility: pllm.Visibility = pllm.Visibility.forward_pinned()
    boundaries: int = 1
    visits: int = 1
    empty: bool = False
    iterations: int = 2
    tokens: str = "all"

    @property
    def weights(self) -> pllm.RequestState:
        return pllm.RequestState({"delta": pllm.Buffer((4, 4)), "tail": pllm.Buffer((), dtype=torch.int64)})

    @pllm.overlay(state="weights", targets=(TARGET,), weight="delta", decode=True, visibility=pllm.Config("visibility"))
    def transition(
        self,
        hidden=pllm.operand("input"),
        desired=pllm.operand("target"),
        delta=pllm.parameter("weights", "delta"),
        tail=pllm.parameter("weights", "tail"),
    ):
        output = hidden @ delta.T
        delta.add_(desired.T @ hidden / len(hidden) ** 2 + 0.01)
        tail.add_(len(hidden))
        count = tail.new_tensor(1)
        return pllm.TensorUpdate(output, count == 2, count, count * len(hidden), {"tail": tail.clone()})

    @pllm.layer_route(tokens="all")
    def route(self, layers: pllm.Layers) -> torch.Tensor:
        if self.empty:
            layers.apply_updates()
        first = layers()
        for _ in range(self.visits - 1):
            layers(start=1, stop=2, hidden=first[0])
        for _ in range(self.boundaries):
            layers.apply_updates()
            if self.empty:
                layers.apply_updates()
            for _ in range(self.iterations):
                layers(start=1, stop=3, hidden=first[0])
        return first[-1]


class UpdatingDecoder(Decoder):
    def layer(self, row, ordinal, positions, hidden, residual, kv="commit"):
        output, carry = super().layer(row, ordinal, positions, hidden, residual)
        if ordinal == TARGET.layer:
            operand = CapturedParameterOperand(
                row.identity,
                TARGET,
                row.phase,
                hidden,
                output,
                torch.eye(4),
                hidden.new_full((), False, dtype=torch.bool),
            )
            output = output + self.overlay.run((operand,))[row.identity, TARGET]
        return output, carry


def oracle(workload):
    cache, outputs = torch.zeros_like(UpdatingDecoder().cache), []
    fields = [torch.zeros(4, 4), torch.zeros(4, 4)]
    tails = [0, 0]
    ordinary = {0: [], 1: []}
    for row in ROWS.rows:
        request, position = row.request_ordinal, row.token_offset
        routed = workload.tokens == "all" or row.phase.value == workload.tokens
        visible = fields[request].clone() if workload.visibility == pllm.Exact() else torch.zeros(4, 4)
        pending = []

        def layer(ordinal, hidden):
            previous = cache[request, ordinal, position - 1] if position else torch.zeros(4)
            cache[request, ordinal, position] = hidden
            base = (hidden * (ordinal + 1) * 0.8 + previous * 0.6).tanh()
            if ordinal == 1:
                result = base + hidden @ visible.T
                if not routed and workload.visibility == pllm.Exact():
                    fields[request] += torch.outer(base, hidden) + 0.01
                    tails[request] += 1
                else:
                    pending.append((hidden.clone(), base.clone()))
                return result
            return base

        hidden = INPUTS[row.packed_row_index].clone()
        first = []
        for ordinal in range(3):
            hidden = layer(ordinal, hidden)
            first.append(hidden.clone())
        outputs.append(hidden)
        if not routed:
            ordinary[request].extend(pending)
            continue
        for _ in range(workload.visits - 1):
            layer(1, first[0])
        for _ in range(workload.boundaries):
            inputs, desired = (torch.stack(tuple(pair[index] for pair in pending)) for index in (0, 1))
            fields[request] += desired.T @ inputs / len(inputs) ** 2 + 0.01
            tails[request] += len(inputs)
            visible = fields[request].clone()
            pending.clear()
            for _ in range(workload.iterations):
                layer(2, layer(1, first[0]))
    for request, pending in ordinary.items():
        if pending:
            inputs, desired = (torch.stack(tuple(pair[index] for pair in pending)) for index in (0, 1))
            fields[request] += desired.T @ inputs / len(inputs) ** 2 + 0.01
            tails[request] += len(inputs)
    return cache, torch.stack(outputs), fields, tails


@pytest.mark.parametrize(
    "boundaries,visits,empty,visibility",
    [
        (0, 1, False, pllm.Visibility.forward_pinned()),
        (1, 1, False, pllm.Visibility.forward_pinned()),
        (2, 2, True, pllm.Visibility.forward_pinned()),
        (2, 2, True, pllm.Exact()),
    ],
)
@pytest.mark.parametrize("tokens", ("all", "prefill", "decode"))
def test_updates_match_independent_private_version_oracle(boundaries, visits, empty, visibility, tokens):
    class Selected(Updating):
        @pllm.layer_route(tokens=tokens)
        def route(self, layers: pllm.Layers) -> torch.Tensor:
            return super().route(layers)

    workload = Selected(visibility, boundaries, visits, empty, tokens=tokens)
    execution = workload.compile_backend(ROUTE_FACTS).execution
    refs = execution.admit(IDENTITIES)
    expected_cache, expected_output, fields, tails = oracle(workload)
    allocations, publications = [], []
    allocate, publish = PageBundleArena.prepare_transition, PageBundleArena.publish

    def allocating(arena, *args, **kwargs):
        result = allocate(arena, *args, **kwargs)
        allocations.append(result)
        return result

    def publishing(arena, write):
        for ref in refs:
            torch.testing.assert_close(execution.snapshot(ref)[0], torch.zeros(4, 4))
        result = publish(arena, write)
        publications.append(result)
        return result

    decoder = UpdatingDecoder()
    with (
        patch.object(PageBundleArena, "prepare_transition", allocating),
        patch.object(PageBundleArena, "publish", publishing),
    ):
        decoder.bind(execution)
        invocation = execution.start_step(VALUE)
        for anchor in range(len(execution.arena.plan.timeline.anchors)):
            execution.execute_anchor(invocation, AnchorId(anchor))
            if anchor == execution.sampler_anchor_id:
                execution.complete_sampler(invocation)
        result = execution.finish_step(invocation)
    torch.testing.assert_close(decoder.output, expected_output)
    torch.testing.assert_close(decoder.cache, expected_cache)
    assert len(allocations) == len(publications) == int(boundaries > 0 or tokens != "all")
    if boundaries or tokens != "all":
        for before, ref in zip(result.publication.before_refs, result.publication.after_refs, strict=True):
            index = refs.index(before)
            delta, tail = execution.snapshot(ref)
            torch.testing.assert_close(delta, fields[index])
            assert tail == tails[index]
        assert sum(int(target.update.trained_rows) for entry in result.entries for target in entry.targets) == sum(
            tails
        )
    assert invocation.sole_write() is None
    execution.retire(IDENTITIES)


def test_marker_source_roundtrip_and_while_rejection():
    assert_finite_resources(Updating(), ROUTE_FACTS)
    graph, _, _ = Updating(empty=True).declare(ROUTE_FACTS)
    assert decode_boundary(encode_boundary(graph), type(graph)) == graph
    assert next(task.replay for task in graph.tasks if task.replay is not None).apply_updates

    class TracedWhile(Updating):
        @pllm.layer_route()
        def route(self, layers: pllm.Layers) -> torch.Tensor:
            return torch.while_loop(lambda hidden: hidden.sum() > 0, lambda hidden: (hidden * 0.5,), (layers.seed,))[0]

    with pytest.raises(ValueError, match="use a bounded for-loop"):
        TracedWhile().declare(ROUTE_FACTS)

    class Unowned(pllm.Workload):
        @pllm.layer_route()
        def route(self, layers: pllm.Layers) -> torch.Tensor:
            layers.apply_updates()
            return layers()[-1]

    with pytest.raises(ValueError, match="one trained parameter overlay"):
        Unowned().declare(ROUTE_FACTS)

    class Second(Updating):
        @pllm.overlay(state="weights", targets=(TARGET,), weight="delta")
        def extra(
            self,
            hidden=pllm.operand("input"),
            desired=pllm.operand("target"),
            delta=pllm.parameter("weights", "delta"),
            tail=pllm.parameter("weights", "tail"),
        ):
            return self.transition(hidden, desired, delta, tail)

    with pytest.raises(ValueError, match="sole request state"):
        Second().declare(ROUTE_FACTS)


@pytest.mark.parametrize("published", (False, True))
def test_cancellation_obeys_the_original_publication_boundary(published):
    execution = Updating().compile_backend(ROUTE_FACTS).execution
    refs = execution.admit(IDENTITIES)
    UpdatingDecoder().bind(execution)
    invocation = execution.start_step(VALUE)
    for anchor in range(3 if published else 2):
        execution.execute_anchor(invocation, AnchorId(anchor))
    receipt = invocation.publication
    execution.cancel_step(invocation)
    assert (receipt is not None) == published
    for ref in refs if receipt is None else receipt.after_refs:
        fields = execution.snapshot(ref)
        assert bool(fields[0].any()) == published
    execution.arena.collect_completed(wait=True)
    assert execution.diagnostics().pending_state_releases == 0
    execution.retire(IDENTITIES)


def test_route_failure_aborts_private_fields():
    execution = Updating().compile_backend(ROUTE_FACTS).execution
    refs = execution.admit(IDENTITIES)
    decoder = UpdatingDecoder()
    decoder.bind(execution)
    invocation = execution.start_step(VALUE)
    layer = decoder.layer

    def fail(row, ordinal, *args):
        if ordinal == 1 and (row, ordinal) in decoder.seen_layers:
            raise RuntimeError("route failed before publication")
        return layer(row, ordinal, *args)

    with patch.object(decoder, "layer", fail), pytest.raises(RuntimeError, match="route failed"):
        execution.execute_anchor(invocation, AnchorId(1))
    for ref in refs:
        assert not bool(execution.snapshot(ref)[0].any())
    execution.arena.collect_completed(wait=True)
    assert execution.diagnostics().pending_state_releases == 0
    execution.retire(IDENTITIES)


def test_untargeted_route_boundary_allocates_and_publishes_nothing():
    class Untargeted(Updating):
        @pllm.layer_route()
        def route(self, layers: pllm.Layers) -> torch.Tensor:
            first = layers(stop=1)
            layers.apply_updates()
            return first[0]

    execution = Untargeted().compile_backend(ROUTE_FACTS).execution
    execution.admit(IDENTITIES)
    UpdatingDecoder().bind(execution)
    invocation = execution.start_step(VALUE)
    for anchor in range(3):
        execution.execute_anchor(invocation, AnchorId(anchor))
    assert invocation.sole_write() is None and invocation.publication is None
    execution.cancel_step(invocation)
    execution.retire(IDENTITIES)
