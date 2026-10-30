"""Route-state authoring, signing, and independent block recurrence checks."""

from dataclasses import dataclass, replace

import pytest
import torch

import pllm
from pllm.config import ExecutionOptions
from pllm.execution.position_state import PositionBuffers
from pllm.execution.token_program import Layers, TokenProgram, TokenTensor
from pllm.workloads.step_loop import StepLoop
from test.workloads.test_recirculation import FACTS, IDENTITIES, VALUE, Decoder, run_step


def test_step_loop_signing_and_provider_agreement():
    workload = StepLoop()
    schema = workload.backend_requirements().position_states
    facts = replace(FACTS, probe=replace(FACTS.probe, position_states=schema))
    deployment = workload.compile_backend(workload.specialize(facts, ExecutionOptions()))
    assert deployment.attachment.plan.semantic_graph.tasks[0].replay.position_states == schema


@pytest.mark.parametrize("case", ("name", "offset", "shape", "dtype"))
def test_positional_authoring_errors(case):
    def route(layers: Layers):
        if case == "name":
            return layers.load("missing")
        if case == "offset":
            return layers.load("history", offset=0)
        layers.save("history", layers.seed[:, :1] if case == "shape" else layers.seed.double())
        return layers.seed

    with pytest.raises(ValueError, match="PositionState|save history"):
        TokenProgram.capture(route, TokenTensor((1, 4), torch.float32, "cpu"), 3, ("history",))


def test_stacked_marker_and_unused_declaration():
    @dataclass(frozen=True)
    class Stacked(pllm.Workload):
        @pllm.layer_route()
        @pllm.write(state="state", fields=("hidden",))
        def route(self, layers):
            return (layers.seed,)

    @dataclass(frozen=True)
    class Unused(pllm.Workload):
        history = pllm.PositionState()

    for workload in (Stacked(), Unused()):
        with pytest.raises(ValueError, match="layer_route"):
            workload.backend_requirements()


@pytest.mark.parametrize("save", (False, True))
def test_ineligible_and_unsaved_tokens_fill_forward(save):
    @dataclass(frozen=True)
    class Fill(pllm.Workload):
        history = pllm.PositionState(initial=2)

        @pllm.layer_route(tokens="decode")
        def route(self, layers):
            if save:
                layers.save("history", layers.load("history").add_(2))
                layers.save("history", layers.load("history") + 1)
            return layers()[-1]

    workload, decoder = Fill(), Decoder()
    facts = replace(FACTS, probe=replace(FACTS.probe, position_states=(("history", 2.0),)))
    execution = workload.compile_backend(workload.specialize(facts, ExecutionOptions())).execution
    execution.admit(IDENTITIES)
    decoder.position = PositionBuffers((torch.full((3, 4), -999.0),), (2.0,), torch.empty(1, 4))
    begin = decoder.begin

    def staged(row):
        current = row.request_ordinal * 2 + row.token_offset // 2
        previous = current - int(row.token_offset > 0 and row.token_offset % 2 == 0)
        decoder.position.indices[0, :, 0] = torch.tensor([current, previous])
        decoder.position.flags[:, 0, 0] = torch.tensor([row.token_offset == 0, row.phase.value == "decode"])
        return begin(row)

    decoder.begin = staged
    run_step(execution, VALUE, decoder)
    torch.testing.assert_close(decoder.position.tables[0], torch.tensor([[2.0] * 4, [2.0] * 4, [2.0 + save] * 4]))
