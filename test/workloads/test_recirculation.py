"""Evidence 27: serial recurrence oracle, row selection and stateless settlement."""

from dataclasses import replace

import pytest
import torch

import pllm
from pllm.config import ExecutionOptions
from pllm.contracts.backend_attachment import (
    AttachmentGraphMode,
    CausalModelDomain,
    ModelProbe,
)
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import (
    AnchorId,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.compile_facts import BackendCompileFacts
from pllm.execution.custom_forward import (
    CustomForwardTask,
    SerialCustomForward,
)
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.step_input import StepInput
from pllm.workload_transport import decode_workload, encode_workload
from pllm.workloads.recirculation import RecirculationWorkload

PROBE = ModelProbe(
    "c" * 64,
    AttachmentGraphMode.EAGER,
    StateDeviceKind.CPU,
    TensorDType.FLOAT32,
    4,
    4,
    (),
    ("model/custom_forward@1",),
    causal_domain=CausalModelDomain(2, 3, 3),
)
FACTS = BackendCompileFacts(
    PROBE, model_timeline(torch.device("cpu"), ("model/custom_forward@1",)), (), 2, 0, 0, layer_count=3
)
WORKLOAD = RecirculationWorkload(alpha=0.7, ramp_tokens=2)
IDENTITIES = (RequestIdentity("prefill", 0, 0), RequestIdentity("decode", 1, 0))
ROWS = PackedPhaseRowBatch(
    1,
    4,
    tuple(
        PackedPhaseRow(
            i,
            int(i == 3),
            i if i < 3 else 0,
            HiddenPhase.PREFILL if i < 3 else HiddenPhase.DECODE,
            IDENTITIES[int(i == 3)],
        )
        for i in range(4)
    ),
)
VALUE = StepInput(ROWS, PackedSamplerRowBatch(ROWS, (PackedSamplerRowLink(0, 2), PackedSamplerRowLink(1, 3))))
INPUTS = torch.tensor([[0.4, -0.1, 0.3, 0.7], [0.2, 0.9, -0.3, 0.8], [-0.2, 0.1, 0.8, 0.4], [0.7, 0.2, -0.1, 0.3]])


class Decoder(SerialCustomForward):
    def __init__(self):
        self.batch = ROWS
        self.cache = torch.zeros(2, 3, 3, 4)
        self.seen_layers = set()
        self.output = None
        self.calls = []

    def bind(self, execution):
        task = next(
            task
            for task in execution.arena.plan.schedule.tasks
            if execution.arena.plan.semantic_graph.tasks[task.task_id].custom_forward is not None
        )
        binding = CustomForwardTask.from_implementation(execution.implementation(task.implementation_id))
        execution.rebind_implementation(replace(binding, primitive=self))

    @property
    def rows(self):
        return self.batch.rows

    def layer(self, row, ordinal, positions, hidden, residual, kv="commit"):
        # A read-only final normalization can legally move after replay.
        # Observe the second call to a source layer, not normalization's
        # position in the reference program.
        if (row, ordinal) in self.seen_layers and ordinal == 1:
            self.calls.append((row.packed_row_index, "replay"))
        self.seen_layers.add((row, ordinal))
        slot = self.cache[row.request_ordinal, ordinal]
        previous = slot[row.token_offset - 1] if row.token_offset else torch.zeros_like(hidden)
        slot[row.token_offset] = hidden
        output = torch.tanh(hidden * (ordinal + 1) * 0.8 + previous * 0.6)
        return output, torch.zeros_like(output)

    def begin(self, row):
        self.calls.append((row.packed_row_index, "normal"))
        return torch.tensor([row.token_offset + int(row.phase is HiddenPhase.DECODE)]), INPUTS[
            row.packed_row_index : row.packed_row_index + 1
        ].clone()

    def normalize(self, row, hidden, residual):
        return hidden + residual

    def commit(self, row, record):
        record()
        self.seen_layers.difference_update((row, layer) for layer in range(self.cache.shape[1]))

    def finish(self, outputs):
        self.output = torch.cat(outputs)


def serial_oracle():
    """Independent serial recurrence over the four packed rows."""
    expected_cache = torch.zeros(2, 3, 3, 4)
    expected_output = []
    for index in range(4):
        request, position = int(index == 3), index if index < 3 else 0
        hidden = INPUTS[index].clone()
        for layer in range(3):
            previous = expected_cache[request, layer, position - 1] if position else torch.zeros(4)
            expected_cache[request, layer, position] = hidden
            hidden = (hidden * (layer + 1) * 0.8 + previous * 0.6).tanh()
            if layer == 0:
                destination = hidden.clone()
        expected_output.append(hidden.clone())
        alpha = 0.7 * min(position + request, 2) / 2
        hidden = (
            destination * (1 - alpha) + hidden * alpha * (destination.square().sum() / hidden.square().sum()).sqrt()
        )
        for layer in (1, 2):
            previous = expected_cache[request, layer, position - 1] if position else torch.zeros(4)
            expected_cache[request, layer, position] = hidden
            hidden = (hidden * (layer + 1) * 0.8 + previous * 0.6).tanh()
    return expected_cache, expected_output


def run_step(execution, value, decoder):
    decoder.bind(execution)
    invocation = execution.start_step(VALUE)
    for anchor in range(8):
        execution.execute_anchor(invocation, AnchorId(anchor))
        if anchor == 3:
            execution.complete_sampler(invocation)
    return invocation, execution.finish_step(invocation)


def test_recirculation_declaration_fits_reference_budget_and_crosses_the_wire() -> None:
    workload = WORKLOAD
    # The exact reference budget, every reservation and task resource at once, admits the plan; one byte less does not.
    reference, _ = workload.specialize(FACTS, ExecutionOptions())
    reference_budget = (
        max(reference.timeline.reserved_bytes)
        + sum(f.workspace_bytes + f.resident_bytes for f in reference.timeline.task_resources)
        + sum(f.bytes for f in reference.timeline.value_resources)
    )
    tight = workload.compile_backend(FACTS, ExecutionOptions(memory_budget_bytes=reference_budget, tuning="measured"))
    assert tight.attachment.plan.timeline.memory_budget_bytes == reference_budget
    with pytest.raises(ValueError):
        workload.specialize(FACTS, ExecutionOptions(memory_budget_bytes=reference_budget - 1))
    restored = decode_workload(encode_workload(workload))
    assert (
        isinstance(restored, RecirculationWorkload) and restored.build(FACTS).finish() == workload.build(FACTS).finish()
    )
    with pytest.raises(ValueError):
        replace(WORKLOAD, destination=3).compile_backend(FACTS)


def test_recirculation_replay_matches_serial_oracle() -> None:
    workload = WORKLOAD
    execution = workload.compile_backend(FACTS, ExecutionOptions(tuning="measured")).execution
    expected_cache, expected_output = serial_oracle()
    assert execution.admit(IDENTITIES) == ()
    rng = torch.random.get_rng_state().clone()
    decoder = Decoder()
    _, published = run_step(execution, VALUE, decoder)
    assert published is None
    torch.testing.assert_close(decoder.output, torch.stack(expected_output))
    torch.testing.assert_close(decoder.cache, expected_cache)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert execution.retire(IDENTITIES) == ()


@pytest.mark.parametrize("tokens, expected", [("prefill", (2, "replay")), ("decode", (3, "replay"))])
def test_route_selects_its_declared_rows(tokens, expected) -> None:
    class Prefill(RecirculationWorkload):
        @pllm.custom_forward(tokens=tokens)
        def recirculate(self, layers: pllm.model.Layers) -> torch.Tensor:
            return super().recirculate(layers)

    decoder = Decoder()
    run_step(Prefill().compile_backend(FACTS).execution, VALUE, decoder)
    assert [call for call in decoder.calls if call[1] == "replay"] == (
        [expected] if tokens == "decode" else [(0, "replay"), (1, "replay"), expected]
    )


def test_route_must_return_a_residual_stream() -> None:
    class MissingOutput(pllm.Workload):
        @pllm.custom_forward()
        def route(self, layers: pllm.model.Layers):
            layers()

    with pytest.raises(ValueError, match="return.*residual"):
        MissingOutput().compile_backend(FACTS)


def test_route_observation_precedes_sampling_normalization():
    from pllm.contracts.semantic_graph import TokenProgramEffect
    from pllm.execution.token_program import TokenProgram, TokenTensor

    decoder, observed = Decoder(), []
    decoder.normalize = lambda row, hidden, residual: 2 * (hidden + residual)
    program = TokenProgram.trace(WORKLOAD.recirculate, TokenTensor((1, 4), torch.float32, "cpu"), 3)
    decoder.run(
        {"prefill": program, "decode": program},
        TokenProgramEffect(3),
        lambda index, hidden: observed.append(hidden.clone()),
        {},
    )
    _, expected = serial_oracle()
    torch.testing.assert_close(torch.cat(observed), torch.stack(expected))
    torch.testing.assert_close(decoder.output, 2 * torch.stack(expected))
