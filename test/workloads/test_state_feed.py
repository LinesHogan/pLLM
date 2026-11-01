"""Public feed declarations and computed writes, independent of the video example."""

from collections import deque
from dataclasses import dataclass, replace
from typing import Literal

import pytest
import torch

import pllm
from pllm.contracts.semantic_graph import SemanticEffect, StateEffectKind, StateResource
from pllm.contracts.state_feed import FeedOrder
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.state_feed import FeedEvent, FeedRecord, select_publications
from pllm.testing import backend_facts

SOURCE = pllm.parameter("source", "value")
TARGET = pllm.parameter("target", "value")


@dataclass(frozen=True)
class FeedWorkload(pllm.Workload):
    """A general tensor feed transformed into a local activation field."""

    order: Literal["ordered", "latest"] = "ordered"
    missing: Literal["stop", "hold"] = "hold"

    @property
    def source(self) -> pllm.InstanceState:
        return pllm.InstanceState({"value": (2,)}, feed=self.order)

    @property
    def target(self) -> pllm.InstanceState:
        return pllm.InstanceState({"value": (2,)})

    @property
    def activation(self) -> pllm.model.ActivationPatch:
        return pllm.model.ActivationPatch(
            "target",
            (0,),
            operator="add",
            decode=True,
            refresh="step",
            on_missing=self.missing,
        )

    @pllm.write(state="target", fields=("value",))
    def transform(self, value: torch.Tensor = SOURCE, previous: torch.Tensor = TARGET) -> tuple[torch.Tensor, ...]:
        """Reading local state and fed state must not be mistaken for two writers."""
        return (value * 2 + previous * 0,)


def facts_for(device=torch.device("cpu")):
    from pllm.contracts.state_representation import StateDeviceKind

    timeline = model_timeline(device, ("transition/full_root@1",))
    provider = PageBundleProviderFacts("test.feed@1", 3, StateDeviceKind(device.type))
    return backend_facts(2, (), timeline, provider, layer_count=1, memory_bytes=65536)


def test_feed_declaration_derives_ownership():
    for order in ("ordered", "latest"):
        workload = FeedWorkload(order)
        assert workload.backend_requirements().activation_layers == (0,)
        deployment = workload.compile_backend(facts_for(), pllm.ExecutionOptions())
        assert deployment.execution.arena.owned_resources == ("target",)
        assert deployment.execution.feeds[0].fields == ("value",)
        plan = deployment.execution.arena.plan
        (group,) = plan.semantic_graph.input_groups
        assert group.borrowed_resources == ("source",)
        assert (
            len(
                [
                    effect
                    for task in plan.semantic_graph.tasks
                    for effect in task.effects
                    if effect.kind is StateEffectKind.TRANSITION_MAYBE
                ]
            )
            == 1
        )


def test_conditional_input_allows_local_reads_but_rejects_an_extra_writer():
    workload = FeedWorkload()
    facts, binding = workload.specialize(facts_for(), pllm.ExecutionOptions())
    binding.graph = replace(
        binding.graph,
        tasks=tuple(
            replace(
                task,
                effects=tuple(
                    SemanticEffect(StateResource("target"), StateEffectKind.TRANSITION)
                    if effect.resource == "target" and effect.kind is StateEffectKind.READ
                    else effect
                    for effect in task.effects
                ),
            )
            if task.symbol == "transform"
            else task
            for task in binding.graph.tasks
        ),
    )
    from pllm.execution.ptts.page_bundle_planning import compile_page_bundle_plan

    with pytest.raises(ValueError, match="one owned transition"):
        compile_page_bundle_plan(
            (binding.graph, binding.layouts), facts.timeline, facts.providers, 2, 65536, facts.device
        )


@pytest.mark.parametrize("order,expected", [(FeedOrder.ORDERED, (1,)), (FeedOrder.LATEST, (1, 2, 3))])
def test_selection_preserves_explicit_missing_and_end_barriers(order, expected):
    queue = deque(
        [
            *(FeedRecord(FeedEvent.PUBLISH, index) for index in (1, 2, 3)),
            FeedRecord(FeedEvent.MISSING),
            FeedRecord(FeedEvent.PUBLISH, 4),
            FeedRecord(FeedEvent.END),
        ]
    )
    assert tuple(record.slot for record in select_publications(queue, order)) == expected
    while queue[0].kind is FeedEvent.PUBLISH:
        select_publications(queue, order)
    assert select_publications(queue, order) == (FeedRecord(FeedEvent.MISSING),)
    assert select_publications(queue, order) == (FeedRecord(FeedEvent.PUBLISH, 4),)
    assert select_publications(queue, order) == (FeedRecord(FeedEvent.END),)


def test_feed_rejects_local_writers_and_request_scopes():
    with pytest.raises(ValueError, match="InstanceState"):
        pllm.RequestState(shape=(2,), feed="ordered")
    with pytest.raises(ValueError, match="optimizer"):
        pllm.InstanceState(shape=(2,), feed="ordered", optimizer=pllm.SGD(0.1))

    @dataclass(frozen=True)
    class WritesSource(FeedWorkload):
        @pllm.write(state="source", fields=("value",))
        def transform(self, value: torch.Tensor = SOURCE, previous: torch.Tensor = TARGET) -> tuple[torch.Tensor, ...]:
            return (value + previous,)

    with pytest.raises(ValueError, match="filled by its feed"):
        WritesSource().build(facts_for())


def test_write_validates_result_shape_during_specialization():
    @dataclass(frozen=True)
    class WrongShape(FeedWorkload):
        @pllm.write(state="target", fields=("value",))
        def transform(self, value: torch.Tensor = SOURCE, previous: torch.Tensor = TARGET) -> tuple[torch.Tensor, ...]:
            return ((value + previous).sum(),)

    with pytest.raises(ValueError, match="write to target needs fields"):
        WrongShape().specialize(facts_for(), pllm.ExecutionOptions())


def test_ordinary_state_ports_and_write_do_not_depend_on_declaration_order():
    from pllm.contracts.packed_rows import PackedPhaseRowBatch, PackedSamplerRowBatch
    from pllm.execution.step_input import StepInput

    @dataclass(frozen=True)
    class LocalWorkload(FeedWorkload):
        @property
        def source(self) -> pllm.InstanceState:
            return pllm.InstanceState({"value": (2,)})

        @property
        def unrelated(self) -> pllm.InstanceState:
            return pllm.InstanceState({"value": (2,)})

        @pllm.write(state="target", fields=("value",))
        def transform(self, value: torch.Tensor = SOURCE, previous: torch.Tensor = TARGET) -> tuple[torch.Tensor, ...]:
            return (torch.ones_like(value) + previous,)

    workload = LocalWorkload()
    execution = workload.compile_backend(facts_for(), pllm.ExecutionOptions()).execution
    execution.admit(())
    rows = PackedPhaseRowBatch(0, 0, ())
    execution.step(StepInput(rows, PackedSamplerRowBatch(rows, ())))
    refs = execution.latest(())
    assert [ref.version_id for ref in refs] == [0, 1, 0]
    torch.testing.assert_close(execution.snapshot(refs[1])[0], torch.ones(2))
    execution.retire(())


def test_write_field_and_feed_field_order_are_signed():
    @dataclass(frozen=True)
    class Fields(FeedWorkload):
        @property
        def target(self) -> pllm.InstanceState:
            return pllm.InstanceState({"value": (2,), "second": (2,)})

        @property
        def activation(self) -> pllm.model.ActivationPatch:
            return pllm.model.ActivationPatch("target", (0, 1), operator="add")

        @pllm.write(state="target", fields=("value",))
        def transform(self, value: torch.Tensor = SOURCE, previous: torch.Tensor = TARGET) -> tuple[torch.Tensor, ...]:
            return (value + previous,)

    first = Fields().build(facts_for()).graph.execution_parameters_digest
    vars(Fields.transform).pop("__pllm_effect__")  # re-declare the same method with another field
    pllm.write(state="target", fields=("second",))(Fields.transform)
    assert Fields().build(facts_for()).graph.execution_parameters_digest != first
    Fields.source = pllm.InstanceState({"value": (2,), "extra": (2,)}, feed="ordered")
    first = Fields().build(facts_for()).graph.execution_parameters_digest
    Fields.source = pllm.InstanceState({"extra": (2,), "value": (2,)}, feed="ordered")
    assert Fields().build(facts_for()).graph.execution_parameters_digest != first


def test_fed_buffer_fields_use_the_same_declared_ports():
    @dataclass(frozen=True)
    class BufferFeed(FeedWorkload):
        @property
        def source(self) -> pllm.InstanceState:
            return pllm.InstanceState({"value": pllm.Field((2,), dtype=torch.float32)}, feed=self.order)

    workload = BufferFeed()
    execution = workload.compile_backend(facts_for(), pllm.ExecutionOptions()).execution
    assert execution.arena.owned_resources == ("target",)
    assert execution.feeds[0].fields == ("value",)
