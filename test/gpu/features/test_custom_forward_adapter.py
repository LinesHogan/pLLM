"""Per-call LoRA oracle for captured loops, compacted decode rows and native prompt spans."""

from collections.abc import Mapping
from dataclasses import dataclass
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.forward_context import override_forward_context

import pllm
from pllm.backends.vllm.adaptive_depth import AdaptiveSlice
from pllm.backends.vllm.custom_forward import VllmCustomForward
from pllm.backends.vllm.parameter_overlay import ProjectionReader
from pllm.contracts.semantic_graph import StateResource
from pllm.execution.parameter_projection import ROWS_PER_BLOCK, projection_blocks
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.token_program_executor import execute_segment
from test.gpu.vllm_adaptive_depth_smoke import read_cells, write_cells

TARGETS = tuple(pllm.model.Projection(name, 23) for name in ("q_proj", "v_proj", "down_proj"))


@dataclass(frozen=True)
class LoopedAdapter(pllm.Workload):
    """Prompt-derived request LoRA, applied to every visit of the final layer."""

    @property
    def lora(self) -> pllm.RequestState:
        return pllm.RequestState(
            {
                "a": pllm.Field(lambda inputs, outputs: (4, inputs)),
                "b": pllm.Field(lambda inputs, outputs: (outputs, 4)),
            },
            init=pllm.Computed(self.initialize),
        )

    def initialize(self, fields: Mapping[str, torch.Tensor], generator: torch.Generator) -> None:
        for value in fields.values():
            value.normal_(std=0.025, generator=generator)

    @property
    def adapter(self) -> pllm.model.Adapter:
        return pllm.model.Adapter("lora", TARGETS, operator="low_rank", scale=0.7)

    @pllm.write(state="lora", fields=("b_0", "b_1", "b_2"))
    def personalize(
        self,
        prompt=pllm.residual(0, tokens="prefill"),
        q=pllm.parameter("lora", "b_0"),
        v=pllm.parameter("lora", "b_1"),
        down=pllm.parameter("lora", "b_2"),
    ):
        gain = 1 + prompt.float().mean(-1).sigmoid()[:, None, None]
        return tuple((value * gain).to(value.dtype) for value in (q, v, down))

    @pllm.custom_forward(tokens="prefill", over="span")
    def prompt(self, layers):
        taps = layers()
        hidden = taps[-2]
        for _ in range(2):
            hidden = layers(23, hidden=hidden)[-1]
        return hidden

    @pllm.custom_forward(tokens="decode")
    def decode(self, layers):
        taps = layers()
        hidden = taps[-2]
        for _ in range(2):
            hidden = layers(23, hidden=hidden, active=layers.position % 3 != 0)[-1]
        return hidden


class PerCallReference:
    """Replace reads with independent request-root matmuls and select active rows on the host, only in the oracle."""

    def __init__(self, owner):
        self.owner = owner
        self.packed = None
        self.fields = {}
        self.checked = 0
        self.compacted = set()
        self.distinct = False
        self.native_replay = torch.cuda.CUDAGraph.replay
        self.native_span = VllmCustomForward.span

    def stage(self):
        context = self.owner.context
        execution = self.owner.execution
        self.fields = {}
        for row in self.owner.rows:
            ((pin, ref),) = context.read_versions((row.identity,), StateResource("lora"), execution.directory)
            (root,) = context.arena.borrow_roots((pin,), StateRefs.of((ref,)))
            self.fields[row.packed_row_index] = tuple(value.clone() for value in root)
        values = tuple(self.fields.values())
        self.distinct |= any(not torch.equal(values[0][1], root[1]) for root in values[1:])

    def emit(self, reader, hidden, desired, pools, table, output, captured, desired_capture):
        # Read authoritative request roots, without using the reader's table or its per-call row scope.
        target = TARGETS.index(reader.binding.target)
        roots = [self.fields[index] for index in self.packed.tolist()]
        a, b = (torch.stack([root[2 * target + field] for root in roots]).to(hidden.dtype) for field in (0, 1))
        delta = (
            self.span_delta(reader, hidden, a, b)
            if len(hidden) > reader.requests
            else torch.bmm(torch.bmm(hidden[:, None], a.transpose(1, 2)), b.transpose(1, 2))[:, 0]
        )
        output[:, reader.offset : reader.offset + reader.width].add_((delta * reader.binding.scale).to(output.dtype))

    def span_delta(self, reader, hidden, a, b):
        """Keep native block GEMM geometry; changing it amplifies BF16 noise through the loop on A100.

        Ownership comes independently from host requests and authoritative roots, without the device read table.
        """
        groups = {}
        for lane, packed in enumerate(self.packed.tolist()):
            groups.setdefault(self.owner.rows[packed].identity, []).append(lane)
        chunks = [
            rows[start : start + ROWS_PER_BLOCK]
            for rows in groups.values()
            for start in range(0, len(rows), ROWS_PER_BLOCK)
        ]
        count = projection_blocks(reader.rows, reader.requests)
        inputs = hidden.new_zeros((count, ROWS_PER_BLOCK, hidden.shape[1]))
        left, right = (value.new_zeros((count, *value.shape[1:])) for value in (a, b))
        for block, lanes in enumerate(chunks):
            inputs[block, : len(lanes)] = hidden[lanes]
            left[block], right[block] = a[lanes[0]], b[lanes[0]]
        products = torch.bmm(torch.bmm(inputs, left.transpose(1, 2)), right.transpose(1, 2))
        delta = hidden.new_empty((len(hidden), b.shape[1]))
        for block, lanes in enumerate(chunks):
            delta[lanes] = products[block, : len(lanes)]
        return delta

    def select(self, selection, step, hidden, active):
        wave = selection.wave
        self.packed = wave.indices[0]
        count = int(active.sum())
        self.compacted.add(count)
        if count == len(hidden):
            return execute_segment(step, hidden, wave.layer)
        selection.propagate(step, hidden, active)
        output = tuple(hidden.clone() for _ in step.taps)
        if count:
            size = next(size for size in selection.widths if count <= size)
            rows = torch.argsort(~active.flatten(), stable=True)[:size]
            self.packed = wave.indices[0].index_select(0, rows)
            selection.compact(step, hidden, active, rows, output)
        self.packed = wave.indices[0]
        return output

    def replay(self, graph):
        wave = next((wave for wave in self.owner.captures.values() if wave.graph is graph), None)
        if wave is None:
            return self.native_replay(graph)
        self.stage()
        saved = read_cells(wave)
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("error")
        try:
            self.native_replay(graph)
        finally:
            torch.cuda.set_sync_debug_mode(mode)
        live = read_cells(wave)
        write_cells(wave, saved)
        self.packed = wave.indices[0]
        observation, results = wave.observation, wave.results
        with (
            override_forward_context(wave.attention.context),
            patch.object(ProjectionReader, "emit", lambda reader, *args: self.emit(reader, *args)),
            patch.object(AdaptiveSlice, "__call__", lambda selection, *args: self.select(selection, *args)),
        ):
            reference = wave.body()
        wave.observation, wave.results = observation, results
        torch.testing.assert_close(wave.output, reference, rtol=0, atol=0)
        for actual, expected in zip(live, read_cells(wave), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        write_cells(wave, live)
        self.checked += 1

    def span(self, owner, rows, program=None):
        self.stage()
        self.native_span(owner, rows, program)
        actual = owner.output.clone()
        self.packed = torch.tensor([row.packed_row_index for row in rows], device=actual.device)
        # Replaying the same positions overwrites only those KV cells, and preserves their causal history.
        with patch.object(ProjectionReader, "emit", lambda reader, *args: self.emit(reader, *args)):
            self.native_span(owner, rows, program)
        torch.testing.assert_close(actual[self.packed], owner.output[self.packed], rtol=0, atol=0)
        owner.output.copy_(actual)


def test_adapter_applies_to_each_custom_forward_projection_call():
    config = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=128,
        max_num_seqs=8,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.3,
        execution=pllm.ExecutionOptions(cuda_graph_pool_budget_bytes=256 << 20),
    )
    report = pllm.explain(LoopedAdapter(), config)
    assert "every projection invocation" in str(report)
    handle = pllm.make_llm(workload=LoopedAdapter(), config=config)
    owner = handle.runner.require_session().custom_forward
    audit = PerCallReference(owner)
    try:
        with (
            patch.object(torch.cuda.CUDAGraph, "replay", lambda graph: audit.replay(graph)),
            patch.object(VllmCustomForward, "span", lambda owner, *args: audit.span(owner, *args)),
        ):
            handle.generate(
                ["The capital of France is" + " very" * i for i in range(8)],
                SamplingParams(max_tokens=5, temperature=0, ignore_eos=True),
            )
        assert audit.checked >= 4
        assert any(0 < count < 8 for count in audit.compacted)
        assert audit.distinct
    finally:
        handle.close()
