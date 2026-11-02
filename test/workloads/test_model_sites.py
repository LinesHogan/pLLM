"""Custom residual hooks on a small CPU decoder with vLLM's norm interface: a stateful memory sublayer against a
serial float64 oracle that walks each request token by token with the chunk rule."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import pllm
from pllm import testing
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch, PackedSamplerRowLink
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.runtime_facts import model_timeline
from pllm.execution.step_input import StepInput

pytest.importorskip("vllm")
WIDTH, LAYERS = 8, 3


class Norm(nn.Module):
    """vLLM's norm interface: normalize, or add the sublayer output into the residual stream and return both."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(WIDTH, dtype=torch.float64) + 0.5)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        stream = x if residual is None else x + residual
        normed = stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + 1e-6) * self.weight
        return normed if residual is None else (normed, stream)


class Layer(nn.Module):
    """A position-independent decoder layer calling its norms as every admitted vLLM family does."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = Norm(), Norm()
        self.attention, self.mlp = nn.Linear(WIDTH, WIDTH).double(), nn.Linear(WIDTH, WIDTH).double()

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        if residual is None:
            residual, hidden = hidden, self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden, residual = self.post_attention_layernorm(torch.tanh(self.attention(hidden)), residual)
        return torch.tanh(self.mlp(hidden)), residual


class Body(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers, self.norm = nn.ModuleList(Layer() for _ in range(LAYERS)), Norm()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual)
        return self.norm(hidden, residual)[0]


@dataclass(frozen=True)
class FastWeightMemory(pllm.Workload):
    """A fast-weight memory beside layer 1's MLP: rows read M k; M learns k -> y once a chunk completes."""

    chunk: int = 2
    lr: float = 0.3

    @property
    def memory(self) -> pllm.RequestState:
        return pllm.RequestState({"M": lambda m: (m.hidden_size, m.hidden_size)}, dtype=torch.float64)

    @pllm.custom_residual(site="mlp", layers=(1,), state="memory", chunk=pllm.Attr("chunk"))
    def remember(self, x, y, M=pllm.parameter("memory", "M")):
        k = torch.nn.functional.normalize(x, dim=-1)
        read = k @ M.T
        return x + y + 0.5 * read, M - self.lr / len(x) * (read - y).T @ k


def oracle(body: Body, workload: FastWeightMemory, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One request's outputs and final memory, token by token: a row reads the memory its chunk started with."""
    memory, start, pending, outputs = torch.zeros(WIDTH, WIDTH, dtype=torch.float64), None, [], []
    for token in tokens:
        start = memory if start is None else start
        hidden, residual = token[None], None
        for index, layer in enumerate(body.layers):
            if residual is None:
                residual, hidden = hidden, layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            hidden, residual = layer.post_attention_layernorm(torch.tanh(layer.attention(hidden)), residual)
            branch = torch.tanh(layer.mlp(hidden))
            if index == 1:
                key = torch.nn.functional.normalize(residual, dim=-1)
                residual, hidden = residual + branch + 0.5 * key @ start.T, torch.zeros_like(branch)
                pending.append((key[0], branch[0]))
            else:
                hidden = branch
        outputs.append(body.norm(hidden, residual)[0][0])
        if len(pending) == workload.chunk:
            keys, values = (torch.stack(column) for column in zip(*pending, strict=True))
            memory = start - workload.lr / workload.chunk * (keys @ start.T - values).T @ keys
            start, pending = None, []
    return torch.stack(outputs), memory


@pytest.mark.parametrize("chunk", (1, 2, 3))
def test_stateful_custom_residual_matches_serial_chunk_oracle(chunk: int) -> None:
    from pllm.backends.vllm.architecture import sublayer_norms
    from pllm.backends.vllm.model_sites import ModelSites

    torch.manual_seed(0)
    body, workload = Body(), FastWeightMemory(chunk=chunk)
    model = SimpleNamespace(model=body)
    decoder = SimpleNamespace(model=model, norms=tuple(norm.forward for norm in sublayer_norms(model)))
    facts = testing.backend_facts(
        WIDTH,
        timeline=model_timeline(torch.device("cpu"), ("transition/full_root@1",)),
        dtype=TensorDType.FLOAT64,
        layer_count=LAYERS,
        memory_bytes=1 << 20,
    )
    execution = testing.compile_execution(workload, facts)
    (rewrite,) = execution.math.model_sites
    sites = ModelSites(decoder, (rewrite,), None, None, 16, 2)
    identities = (testing.RequestIdentity("a", 0, 0), testing.RequestIdentity("b", 1, 0))
    refs = execution.admit(identities)
    prompts = (torch.randn(5, WIDTH, dtype=torch.float64), torch.randn(3, WIDTH, dtype=torch.float64))
    decoded = torch.randn(4, 2, WIDTH, dtype=torch.float64)
    # Steps: the prompts split across two steps (chunked prefill), then four decode steps of one token each.
    steps = (
        ((0, 0, 3), (1, 0, 2)),
        ((0, 3, 5), (1, 2, 3)),
        *(((0, 5 + t, 6 + t), (1, 3 + t, 4 + t)) for t in range(4)),
    )
    outputs: dict[int, list[torch.Tensor]] = {0: [], 1: []}
    for step_id, spans in enumerate(steps):
        rows, inputs = [], []
        for request, begin, end in spans:
            for offset in range(begin, end):
                length = len(prompts[request])
                phase = HiddenPhase.PREFILL if offset < length else HiddenPhase.DECODE
                rows.append(PackedPhaseRow(len(rows), request, offset, phase, identities[request], length))
                inputs.append(prompts[request][offset] if offset < length else decoded[offset - length, request])
        batch = PackedPhaseRowBatch(step_id, len(rows), tuple(rows))
        links = tuple(
            PackedSamplerRowLink(request, max(row.packed_row_index for row in rows if row.request_ordinal == request))
            for request in (0, 1)
        )
        context = execution.start_step(StepInput(batch, PackedSamplerRowBatch(batch, links)))
        execution.advance_to(context, "request.prepare")
        sites.prepare(batch, context, execution.directory)
        result = body(torch.stack(inputs))
        for anchor in execution.arena.plan.timeline.anchors[1:]:
            execution.advance_to(context, anchor.symbol)
            if anchor.ordinal == execution.sampler_anchor_id:
                execution.complete_sampler(context)
        execution.finish_step(context)
        for row, value in zip(rows, result, strict=True):
            outputs[row.request_ordinal].append(value)
    sites.close()
    for request in (0, 1):
        tokens = torch.cat((prompts[request], decoded[:, request]))
        expected, memory = oracle(body, workload, tokens)
        torch.testing.assert_close(torch.stack(outputs[request]), expected, rtol=1e-12, atol=1e-12)
        latest = execution.arena.latest(refs[request].lineage_id)
        torch.testing.assert_close(execution.snapshot(latest)[0], memory, rtol=1e-12, atol=1e-12)
