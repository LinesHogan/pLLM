"""CPU FP64 chunk oracle: declared SGD/AdamW, causal KV, tail, first decode token and zero identity."""

from dataclasses import dataclass, replace

import torch

import pllm
from pllm.authoring import ParameterTargetShape, TensorStepInput
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.trace import TraceSelection
from pllm.workloads.prefill_adapters import PrefillAdapters
from test.backend_facts import backend_facts


def zeros(fields, generator):
    for value in fields.values():
        value.zero_()


@dataclass(frozen=True)
class Adapters(PrefillAdapters):
    optimizer: pllm.SGD | pllm.AdamW = pllm.SGD(1e-3)
    zero: bool = False

    @property
    def adapters(self) -> pllm.RequestState:
        state = super().adapters
        return replace(state, optimizer=self.optimizer, initialize=zeros if self.zero else state.initialize)


class Model:
    def __init__(self, tokens):
        rng = torch.Generator().manual_seed(71)
        rand = lambda *shape: torch.randn(shape, generator=rng, dtype=torch.float64) * 0.2
        self.layers = tuple(tuple(rand(4, 4) for _ in range(3)) + (rand(6, 4), rand(4, 6)) for _ in range(3))
        self.embedding, self.head = rand(17, 4), rand(17, 4)
        self.tokens, self.length = tokens, len(tokens)
        self.kv = [(torch.empty(0, 4, dtype=torch.float64),) * 2 for _ in self.layers]
        self.history = []
        self.weight = self.layers[0][-1]

    @torch.no_grad()
    def layer(self, layer, hidden, history):
        q, k, v, up, down = self.layers[layer]
        keys, values = (torch.cat((past, hidden @ weight.T)) for past, weight in zip(history, (k, v)))
        mask = torch.arange(len(keys))[None, :] > len(history[0]) + torch.arange(len(hidden))[:, None]
        target = hidden + (hidden @ q.T @ keys.T / 2).masked_fill(mask, -torch.inf).softmax(-1) @ values
        return torch.tanh(target @ up.T), target, down, (keys, values)

    def forward(self, start, stop, parameters):
        h = self.embedding[self.tokens[start:stop]]
        for layer in range(len(self.kv)):
            x, target, weight, self.kv[layer] = self.layer(layer, h, self.kv[layer])
            output = x @ weight.T
            if layer == 0:
                output = output + (x @ parameters[0].T) @ parameters[1].T
                local = x, target
            h = target + output
        self.history.append(tuple(t.clone() for kv in self.kv for t in kv))
        return h, *local

    def finish(self, outputs):
        self.output = torch.cat(outputs)


def execute(deployment, backend, phase):
    owner = deployment.execution
    offset = len(backend.kv[0][0])
    backend.row = PackedPhaseRow(0, 0, offset, phase, RequestIdentity("oracle", 0, 0))
    rows = PackedPhaseRowBatch(
        1,
        offset,
        backend.length,
        tuple(replace(backend.row, packed_row_index=i, token_offset=offset + i) for i in range(backend.length)),
    )
    invocation = owner.start_step(TensorStepInput(rows, PackedSamplerRowBatch(rows, ())))
    invocation.context.model_scan = backend
    owner.execute_anchor(invocation, deployment.attachment.inputs.request.task_bindings[0].anchor_id)
    owner.execute_anchor(invocation, owner.sampler_anchor_id)
    owner.complete_sampler(invocation)
    owner.finish_step(invocation)


def test_model_scan_matches_independent_cpu_oracle_bitwise():
    torch.set_num_threads(1)
    for config, zero in ((pllm.SGD(1e-3), False), (pllm.AdamW(1e-3), False), (pllm.SGD(1e-3), True)):
        workload = Adapters(chunk_size=4, rank=2, dtype=torch.float64, optimizer=config, zero=zero)
        (target,) = workload.backend_requirements().parameter_targets
        facts = backend_facts(
            4,
            (),
            RuntimeFacts(torch.device("cpu")).timeline(("model/chunk_scan@1", "transition/full_root@1")),
            PageBundleProviderFacts("test.model_scan@1", 5),
            max_lineages=1,
            memory_bytes=1 << 20,
            max_rows=11,
            dtype=TensorDType.FLOAT64,
            targets=(ParameterTargetShape(target, 6, 4),),
            layer_count=3,
        )
        facts = replace(facts, probe=replace(facts.probe, capabilities=("model/chunk_scan@1",)))
        deployment = workload.compile_backend(workload.specialize(facts, pllm.ExecutionOptions()))
        owner = deployment.execution
        owner.trace.configure(TraceSelection(tensors=True))
        (ref,) = owner.admit((RequestIdentity("oracle", 0, 0),))
        fields = deployment.attachment.inputs.request.overlay_bindings[0].fields
        params = [owner.snapshot(ref)[i].requires_grad_() for i in fields]
        optimizer = (torch.optim.AdamW if isinstance(config, pllm.AdamW) else torch.optim.SGD)(
            params, lr=config.lr, weight_decay=0, foreach=False
        )
        tokens = torch.tensor([1, 4, 2, 7, 3, 6, 11, 8, 0, 5, 9])
        backend = Model(tokens)
        kv = [(torch.empty(0, 4, dtype=torch.float64),) * 2 for _ in backend.layers]
        expected = {}
        history, outputs = [], []
        for chunk, start in enumerate(range(0, len(tokens), 4)):
            with torch.no_grad():
                h = backend.embedding[tokens[start : start + 4]]
                for layer in range(len(kv)):
                    x, target, weight, kv[layer] = backend.layer(layer, h, kv[layer])
                    out = x @ weight.T
                    if layer == 0:
                        local = x, target, weight
                        if not zero:
                            out = out + (x @ params[0].T) @ params[1].T
                    h = target + out
            x, target, weight = local
            prediction = x @ weight.T + (x @ params[0].T) @ params[1].T
            dtype = torch.promote_types(prediction.dtype, torch.float32)
            gradients = torch.autograd.grad((prediction.to(dtype) - target.to(dtype)).square().mean(), params)
            for p, g in zip(params, gradients):
                p.grad = g
            optimizer.step()
            expected.update(
                {(chunk, name): value.detach().clone() for name, value in zip(("hidden", "a", "b"), (h, *params))}
            )
            outputs.append(h)
            history.append(tuple(t.clone() for pair in kv for t in pair))
        execute(deployment, backend, HiddenPhase.PREFILL)
        actual = [backend.output, *(t for step in backend.history for t in step)]
        reference = [torch.cat(outputs), *(t for step in history for t in step)]
        events = [event for event in owner.trace.snapshot().events if event.tensor_name]
        actual += [event.tensor() for event in events]
        reference += [expected[event.iteration, event.tensor_name] for event in events]
        for got, want in zip(actual, reference, strict=True):
            assert torch.equal(got.contiguous().view(torch.int64), want.contiguous().view(torch.int64))
        assert owner.diagnostics().update_count == 3
        assert owner.arena.latest(ref.lineage_id).version_id == 3
        tokens = (reference[0][-1] @ backend.head.T).argmax().reshape(1)
        assert torch.equal(tokens, (backend.output[-1] @ backend.head.T).argmax().reshape(1))
        owner.retire((backend.row.identity,))
        assert owner.diagnostics().active_lineages == owner.diagnostics().pending_state_releases == 0
