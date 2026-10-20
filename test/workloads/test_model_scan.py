"""Independent serial FP64 oracle for the complete five-target ModelScan grid."""

import json
import re
from dataclasses import dataclass, replace

import pytest
import torch

import pllm
from pllm.authoring import ParameterTargetShape, TensorStepInput
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.region_compilation import analyze_region_program, validate_region_schedule
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.tensor_borrow import TensorResourceView
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
    layers: tuple[int, ...] = (0, 2, 4, 6, 8)

    @property
    def adapters(self) -> pllm.RequestState:
        state = super().adapters
        return replace(state, optimizer=self.optimizer, initialize=zeros if self.zero else state.initialize)

    @property
    def prefill(self) -> pllm.ModelScan:
        scan = super().prefill
        return replace(
            scan, apply=replace(scan.apply, targets=tuple(t for t in scan.apply.targets if t.layer in self.layers))
        )


class Model:
    def __init__(self, tokens):
        rng = torch.Generator().manual_seed(71)
        rand = lambda *shape: torch.randn(shape, generator=rng, dtype=torch.float64) * 0.2
        self.layers = tuple(tuple(rand(4, 4) for _ in range(3)) + (rand(6, 4), rand(4, 6)) for _ in range(10))
        self.embedding, self.head = rand(17, 4), rand(17, 4)
        self.tokens, self.length = tokens, len(tokens)
        self.kv = [torch.zeros(2, 32, 4, dtype=torch.float64) for _ in self.layers]
        self.lengths = [0] * len(self.layers)
        self.history = {}

    def weight(self, layer):
        return self.layers[layer][-1]

    def inputs(self, chunk):
        return tuple(
            pad(self.embedding[self.tokens[start : start + chunk]], chunk) for start in range(0, self.length, chunk)
        )

    def kv_view(self, resource):
        return TensorResourceView(resource, tuple(self.kv))

    @torch.no_grad()
    def layer(self, layer, hidden, history):
        q, k, v, up, down = self.layers[layer]
        keys, values = (torch.cat((past, hidden @ weight.T)) for past, weight in zip(history, (k, v)))
        mask = torch.arange(len(keys))[None, :] > len(history[0]) + torch.arange(len(hidden))[:, None]
        target = hidden + (hidden @ q.T @ keys.T / 2).masked_fill(mask, -torch.inf).softmax(-1) @ values
        return torch.tanh(target @ up.T), target, down, (keys, values)

    def forward(self, layer, start, rows, hidden, residual, parameters):
        prefix = self.lengths[layer]
        x, target, weight, kv = self.layer(layer, hidden[:rows], self.kv[layer][:, :prefix])
        with torch.no_grad():
            out = x @ weight.T
            if parameters:
                out = out + (x @ parameters[0].T) @ parameters[1].T
            h = target + out
            self.kv[layer][:, : prefix + rows] = torch.stack(kv)
        self.lengths[layer] += rows
        self.history[layer, start] = torch.stack(kv)
        h, r = pad(h, len(hidden)), torch.zeros_like(hidden)
        return (h, r, pad(x, len(hidden)), pad(target, len(hidden))) if parameters else (h, r)

    def finish(self, outputs, residuals, chunk):
        self.output = torch.cat(outputs)[: self.length]


def pad(value, rows):
    return torch.nn.functional.pad(value, (0, 0, 0, rows - len(value)))


def execute(deployment, backend, phase):
    owner = deployment.execution
    offset = backend.lengths[0]
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
        targets = workload.backend_requirements().parameter_targets
        facts = backend_facts(
            4,
            (),
            RuntimeFacts(torch.device("cpu")).timeline(("model/chunk_scan@1", "transition/full_root@1")),
            PageBundleProviderFacts("test.model_scan@1", 5),
            max_lineages=1,
            memory_bytes=1 << 24,
            max_rows=11,
            dtype=TensorDType.FLOAT64,
            targets=tuple(ParameterTargetShape(t, 6, 4) for t in targets),
            layer_count=10,
        )
        facts = replace(facts, probe=replace(facts.probe, capabilities=("model/chunk_scan@1",)))
        deployment = workload.compile_backend(workload.specialize(facts, pllm.ExecutionOptions()))
        owner = deployment.execution
        owner.trace.configure(TraceSelection(tensors=True, max_events=4096))
        refs = owner.admit((RequestIdentity("oracle", 0, 0),))
        bindings = deployment.attachment.inputs.request.overlay_bindings
        params = {
            binding.target.layer: [owner.snapshot(ref)[i].requires_grad_() for i in binding.fields]
            for ref, binding in zip(refs, bindings, strict=True)
        }
        if config == pllm.SGD(1e-3) and not zero:
            subset = replace(workload, layers=(2, 4, 6, 8))
            smaller = subset.compile_backend(
                subset.specialize(replace(facts, targets=facts.targets[1:]), pllm.ExecutionOptions())
            )
            other = RequestIdentity("other", 0, 0)
            smaller.execution.admit((other,))
            smaller.execution.retire((other,))
            subset_refs = smaller.execution.admit((RequestIdentity("oracle", 0, 0),))
            for target, ref in zip(subset.layers, subset_refs, strict=True):
                for got, want in zip(smaller.execution.snapshot(ref), params[target], strict=True):
                    assert torch.equal(got, want)
            smaller.execution.retire((RequestIdentity("oracle", 0, 0),))
        optimizers = {
            layer: (torch.optim.AdamW if isinstance(config, pllm.AdamW) else torch.optim.SGD)(
                values, lr=config.lr, weight_decay=0, foreach=False
            )
            for layer, values in params.items()
        }
        tokens = torch.tensor([1, 4, 2, 7, 3, 6, 11, 8, 0, 5, 9])
        backend = Model(tokens)
        kv = [(torch.empty(0, 4, dtype=torch.float64),) * 2 for _ in backend.layers]
        expected, history, outputs = {}, {}, []
        for chunk, start in enumerate(range(0, len(tokens), 4)):
            h = backend.embedding[tokens[start : start + 4]]
            r = h
            for layer in range(len(kv)):
                record = {"input": pad(h, 4), "residual_in": pad(r, 4)}
                x, target, weight, kv[layer] = backend.layer(layer, h, kv[layer])
                with torch.no_grad():
                    out = x @ weight.T
                    if layer in params:
                        a, b = params[layer]
                        record.update(a_read=a.clone(), b_read=b.clone())
                        if not zero:
                            out = out + (x @ a.T) @ b.T
                    h = target + out
                r = torch.zeros_like(h)
                record.update(
                    hidden=pad(h, 4),
                    residual=pad(r, 4),
                    x=pad(x if layer in params else h, 4),
                    target=pad(target if layer in params else h, 4),
                )
                if layer in params:
                    a, b = params[layer]
                    prediction = x @ weight.T + (x @ a.T) @ b.T
                    dtype = torch.promote_types(prediction.dtype, torch.float32)
                    a.grad, b.grad = torch.autograd.grad(
                        (prediction.to(dtype) - target.to(dtype)).square().mean(), (a, b)
                    )
                    optimizers[layer].step()
                    record.update(a=a.detach().clone(), b=b.detach().clone())
                expected.update(((chunk, f"{layer}.{name}"), value) for name, value in record.items())
                history[layer, start] = torch.stack(kv[layer])
            outputs.append(h)
        execute(deployment, backend, HiddenPhase.PREFILL)
        events = [event for event in owner.trace.snapshot().events if event.tensor_name]
        pairs = [
            (backend.output, torch.cat(outputs)),
            *((backend.history[key], value) for key, value in history.items()),
            *((event.tensor(), expected[event.iteration, event.tensor_name]) for event in events),
        ]
        for got, want in pairs:
            assert torch.equal(got.contiguous().view(torch.int64), want.contiguous().view(torch.int64))
        trace = [
            (event.iteration, int(event.tensor_name.split(".")[0]))
            for event in events
            if event.tensor_name.endswith(".input")
        ]
        assert trace != sorted(trace) and set(trace) == {(c, l) for c in range(3) for l in range(10)}
        for event in events:
            if event.tensor_name.endswith("a_read"):
                assert len(event.refs) == 1 and event.refs[0].version_id == event.iteration
        assert owner.diagnostics().update_count == 15
        assert all(owner.arena.latest(ref.lineage_id).version_id == 3 for ref in refs)
        first = (torch.cat(outputs)[-1] @ backend.head.T).argmax()
        assert torch.equal(first, (backend.output[-1] @ backend.head.T).argmax())
        backend.tokens, backend.length = first.reshape(1), 1
        h = backend.embedding[backend.tokens]
        for layer in range(len(kv)):
            x, target, weight, kv[layer] = backend.layer(layer, h, kv[layer])
            with torch.no_grad():
                out = x @ weight.T
                if layer in params and not zero:
                    a, b = params[layer]
                    out = out + (x @ a.T) @ b.T
                h = target + out
        execute(deployment, backend, HiddenPhase.DECODE)
        assert torch.equal(backend.output.view(torch.int64), h.view(torch.int64))
        assert owner.diagnostics().update_count == 15
        source = owner.arena.plan.semantic_graph.source_regions[0].program
        analysis = analyze_region_program(source)
        # Reverse all statement times: complete domain, deliberately illegal causal order.
        illegal = re.sub(
            r"-> \[([^\]]+)\]",
            lambda match: "-> [" + ", ".join("-(" + term.split("=")[-1] + ")" for term in match[1].split(",")) + "]",
            analysis.reference_schedule,
        )
        with pytest.raises(ValueError, match="violates an original value or resource dependency"):
            validate_region_schedule(source, illegal)
        print(json.dumps({"optimizer": type(config).__name__, "zero": zero, "compared": len(pairs), "trace": trace}))
        owner.retire((backend.row.identity,))
