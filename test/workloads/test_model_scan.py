"""Independent serial FP64 oracle for the complete five-target ModelScan grid."""

import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import product

import pytest
import torch

import pllm
from pllm.authoring import ParameterTargetShape, TensorStepInput
from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch, PackedSamplerRowBatch
from pllm.contracts.region_imports import KernelAllocation, NativeKernelImport
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.tensor_layout import TensorDType
from pllm.execution.model_scan import ModelScanSpan
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts
from pllm.execution.region_compilation import analyze_region_program, validate_region_schedule
from pllm.execution.runtime_facts import RuntimeFacts
from pllm.execution.tensor_borrow import PagedAllocationView, ReadonlyTensorCell, ReadonlyTensorView
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
        self.prompt_length = len(tokens)
        self.kv = [torch.zeros(2, 32, 4, dtype=torch.float64) for _ in self.layers]
        self.lengths = [0] * len(self.layers)
        self.history = {}

    def weight(self, layer):
        return self.layers[layer][-1]

    def inputs(self, request, chunk):
        span = self.spans[request]
        begin = span.row.token_offset if span.learning else 0
        return tuple(
            pad(self.embedding[self.tokens[start : start + min(chunk, begin + span.length - start)]], chunk)
            for start in range(begin, begin + span.length, chunk)
        )

    def kv_view(self, resource):
        return PagedAllocationView(resource, tuple(cache.unsqueeze(0) for cache in self.kv), ((0,),))

    def weights_view(self, resource):
        return ReadonlyTensorView(
            resource, tuple(ReadonlyTensorCell.bind((i,), weights) for i, weights in enumerate(self.layers))
        )

    def input_view(self, resource):
        return ReadonlyTensorView(
            resource,
            tuple(
                ReadonlyTensorCell.bind(indices, (self.tokens, self.embedding) if indices[0] == 0 else ())
                for indices in product(*(range(n) for n in resource.domain))
            ),
        )

    def borrow_streams(self, owner, lanes):
        raise AssertionError("CPU reference backend has no CUDA streams")

    @torch.no_grad()
    def layer(self, layer, hidden, history):
        q, k, v, up, down = self.layers[layer]
        keys, values = (torch.cat((past, hidden @ weight.T)) for past, weight in zip(history, (k, v)))
        mask = torch.arange(len(keys))[None, :] > len(history[0]) + torch.arange(len(hidden))[:, None]
        target = hidden + (hidden @ q.T @ keys.T / 2).masked_fill(mask, -torch.inf).softmax(-1) @ values
        return torch.tanh(target @ up.T), target, down, (keys, values)

    def forward(self, request, layer, start, rows, hidden, residual, parameters):
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

    def finish(self, outputs, residuals, chunks):
        self.output = torch.cat(outputs[0])[: self.spans[0].length]


def pad(value, rows):
    return torch.nn.functional.pad(value, (0, 0, 0, rows - len(value)))


def execute(deployment, backend, phase):
    owner = deployment.execution
    window = owner.arena.plan.schedule_physical.source_schedules[0].window
    limit = window.chunks * window.request.quantum if phase is HiddenPhase.PREFILL else 1
    outputs = []
    for start in range(0, backend.length, limit):
        offset = backend.lengths[0]
        length = min(limit, backend.length - start)
        backend.row = PackedPhaseRow(0, 0, offset, phase, RequestIdentity("oracle", 0, 0))
        backend.spans = (ModelScanSpan(backend.row, length, backend.prompt_length),)
        rows = PackedPhaseRowBatch(
            1,
            offset,
            length,
            tuple(replace(backend.row, packed_row_index=i, token_offset=offset + i) for i in range(length)),
        )
        invocation = owner.start_step(TensorStepInput(rows, PackedSamplerRowBatch(rows, ())))
        invocation.context.model_scan = backend
        owner.execute_anchor(invocation, deployment.attachment.inputs.request.task_bindings[0].anchor_id)
        owner.execute_anchor(invocation, owner.sampler_anchor_id)
        owner.complete_sampler(invocation)
        owner.finish_step(invocation)
        outputs.append(backend.output)
    backend.output = torch.cat(outputs)


def imported_facts(workload, facts):
    """The fixed toy backend's bounded CPU layer contract; it is never a CUDA kernel."""
    graph, _, _ = workload.declare(facts)
    imports = tuple(
        NativeKernelImport(
            function,
            "test.toy_causal_layer@1",
            "cpu",
            (KernelAllocation("temporary_tensors", "cpu", 1 << 16),),
            host_reads=True,
        )
        for function in graph.source_regions[0].program.functions
        if not function.name.startswith("train.")
    )
    return replace(facts, probe=replace(facts.probe, region_imports=imports))


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
        facts = replace(
            facts,
            probe=replace(
                facts.probe, capabilities=("model/chunk_scan@1",), causal_domain=CausalModelDomain(1, 10, 32)
            ),
        )
        facts = imported_facts(workload, facts)
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
                subset.specialize(
                    imported_facts(subset, replace(facts, targets=facts.targets[1:])), pllm.ExecutionOptions()
                )
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
        kv, expected, history, outputs = serial(backend, params, optimizers, zero)
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
        if config == pllm.SGD(1e-3) and not zero:
            multiple_requests(workload, facts)


def serial(backend, params, optimizers, zero):
    tokens = backend.tokens
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
                a.grad, b.grad = torch.autograd.grad((prediction.to(dtype) - target.to(dtype)).square().mean(), (a, b))
                optimizers[layer].step()
                record.update(a=a.detach().clone(), b=b.detach().clone())
            expected.update(((chunk, f"{layer}.{name}"), value) for name, value in record.items())
            history[layer, start] = torch.stack(kv[layer])
        outputs.append(h)
    return kv, expected, history, outputs


class Batch:
    """The toy model's request pages stay fixed while packed admission order changes."""

    def __init__(self, models):
        self.models = models
        self.caches = tuple(torch.stack([m.kv[layer] for m in models.values()]) for layer in range(10))
        self.slots = {identity: i for i, identity in enumerate(models)}
        for identity, model in models.items():
            model.kv = [cache[self.slots[identity]] for cache in self.caches]
        self.outputs = {identity: [] for identity in models}

    def weight(self, layer):
        return next(iter(self.models.values())).weight(layer)

    def inputs(self, request, chunk):
        span = self.spans[request]
        model = self.models[span.row.identity]
        model.spans = (replace(span, row=replace(span.row, packed_row_index=0, request_ordinal=0)),)
        return model.inputs(0, chunk)

    def kv_view(self, resource):
        return PagedAllocationView(resource, self.caches, tuple((self.slots[s.row.identity],) for s in self.spans))

    def weights_view(self, resource):
        return next(iter(self.models.values())).weights_view(resource)

    def input_view(self, resource):
        return ReadonlyTensorView(
            resource,
            tuple(ReadonlyTensorCell.bind(indices, ()) for indices in product(*(range(n) for n in resource.domain))),
        )

    def forward(self, request, *args):
        return self.models[self.spans[request].row.identity].forward(0, *args)

    def finish(self, outputs, residuals, chunks):
        for span, tensors in zip(self.spans, outputs, strict=True):
            self.outputs[span.row.identity].append(torch.cat(tensors)[: span.length])


def multiple_requests(workload, facts):
    """Compare variable prompts across reordered cuts and reuse retired request capacity."""
    facts = replace(
        facts,
        max_admitted_lineages=3,
        providers=(PageBundleProviderFacts("test.model_scan@1", 12),),
        probe=replace(facts.probe, causal_domain=CausalModelDomain(3, 10, 32)),
    )
    facts = imported_facts(workload, facts)
    deployment = workload.compile_backend(workload.specialize(facts, pllm.ExecutionOptions()))
    from pllm.execution.region_streams import plan_closed_window

    source = deployment.attachment.inputs.plan.semantic_graph.source_regions[0]
    with pytest.raises(ValueError, match="increase the memory budget"):
        plan_closed_window(source, facts.timeline.region_policy, parallel=False, budget=1)
    with pytest.raises(ValueError, match="engine token budget"):
        plan_closed_window(
            replace(source, window=replace(source.window, token_budget=1)), facts.timeline.region_policy, parallel=False
        )
    owner, bindings = deployment.execution, deployment.attachment.inputs.request.overlay_bindings
    identities = tuple(RequestIdentity(str(i), 0, 0) for i in range(3))
    models = {identity: Model(torch.arange(length) % 17) for identity, length in zip(identities, (3, 9, 11))}
    expected, roots = {}, {}
    for identity, model in models.items():
        refs = owner.admit((identity,))
        roots[identity] = refs
        params = {
            binding.target.layer: [owner.snapshot(ref)[i].requires_grad_() for i in binding.fields]
            for ref, binding in zip(refs, bindings, strict=True)
        }
        optimizers = {layer: torch.optim.SGD(values, lr=1e-3, foreach=False) for layer, values in params.items()}
        expected[identity] = serial(model, params, optimizers, False)
    batch = Batch(models)
    owner.trace.configure(TraceSelection(tensors=True, max_events=4096))
    window = owner.arena.plan.schedule_physical.source_schedules[0].window
    limit = window.chunks * window.request.quantum
    pending = list(identities)
    rounds = 0
    while pending:
        pending.reverse()
        spans, rows = [], []
        for identity in pending:
            model = models[identity]
            offset = model.lengths[0]
            length = min(limit, model.length - offset)
            if len(rows) + length > facts.probe.max_packed_rows:
                continue
            row = PackedPhaseRow(len(rows), len(spans), offset, HiddenPhase.PREFILL, identity)
            spans.append(ModelScanSpan(row, length, model.length))
            rows.extend(
                replace(row, packed_row_index=row.packed_row_index + i, token_offset=offset + i) for i in range(length)
            )
        batch.spans = tuple(spans)
        packed = PackedPhaseRowBatch(1, rounds, len(rows), tuple(rows))
        invocation = owner.start_step(TensorStepInput(packed, PackedSamplerRowBatch(packed, ())))
        invocation.context.model_scan = batch
        owner.execute_anchor(invocation, deployment.attachment.inputs.request.task_bindings[0].anchor_id)
        owner.execute_anchor(invocation, owner.sampler_anchor_id)
        owner.complete_sampler(invocation)
        owner.finish_step(invocation)
        for identity in tuple(pending):
            if models[identity].lengths[0] == models[identity].length:
                count = (models[identity].length + 3) // 4
                assert all(owner.arena.latest(ref.lineage_id).version_id == count for ref in roots[identity])
                owner.retire((identity,))
                pending.remove(identity)
        rounds += 1
    assert owner.diagnostics().update_count == 5 * sum((model.length + 3) // 4 for model in models.values())
    comparisons = 0
    for identity, model in models.items():
        kv, records, history, outputs = expected[identity]
        pairs = [(torch.cat(batch.outputs[identity]), torch.cat(outputs))]
        pairs.extend((model.history[key], value) for key, value in history.items())
        pairs.extend(
            (event.tensor(), records[event.iteration, event.tensor_name])
            for event in owner.trace.snapshot().events
            if event.tensor_name and event.row.identity == identity
        )
        for got, want in pairs:
            assert torch.equal(got.contiguous().view(torch.int64), want.contiguous().view(torch.int64))
        comparisons += len(pairs)
    fresh = RequestIdentity("fresh", 1, 0)
    refs = owner.admit((fresh,))
    assert all(ref.version_id == 0 for ref in refs)
    owner.retire((fresh,))
    assert owner.diagnostics().active_lineages == owner.diagnostics().pending_state_releases == 0
    print(json.dumps(dict(multi_request_comparisons=comparisons, rounds=rounds)))
    cancel_windows(deployment)


def cancel_windows(deployment):
    """Faults preserve only completed publications and release every request pin."""
    owner = deployment.execution
    for point in ("forward", "update", "publish", "tail"):
        identity = RequestIdentity(point, 2, 0)
        refs = owner.admit((identity,))
        model = Model(torch.tensor([1, 2, 3, 4, 5]))
        row = PackedPhaseRow(0, 0, 0, HiddenPhase.PREFILL, identity)
        model.spans = (ModelScanSpan(row, 5, 5),)
        rows = PackedPhaseRowBatch(1, 0, 5, tuple(replace(row, packed_row_index=i, token_offset=i) for i in range(5)))
        invocation = owner.start_step(TensorStepInput(rows, PackedSamplerRowBatch(rows, ())))
        context = invocation.context
        context.model_scan = model
        with inject_fault(point, model, lambda: context, 4) as (fired, _):
            with pytest.raises(RuntimeError):
                owner.execute_anchor(invocation, deployment.attachment.inputs.request.task_bindings[0].anchor_id)
            if context.release is None:
                owner.cancel_step(invocation)
        assert fired == [point]
        versions = tuple(owner.arena.latest(ref.lineage_id).version_id for ref in refs)
        if point in ("forward", "update"):
            assert not any(versions)
        elif point == "publish":
            assert sum(versions) == 1
        else:
            assert any(versions)
        owner.retire((identity,))
        usage = owner.arena.usage()
        assert usage.private_versions == usage.pending_receipts == owner.arena.active_lineages == 0
        print(json.dumps(dict(cancel=point, versions=versions)))


@contextmanager
def inject_fault(point, model, context, chunk):
    """Cancel after real work; the pending case delays only the fault-injection stream."""
    from unittest.mock import patch

    from pllm.execution.model_scan import ModelScanInvocation
    from pllm.execution.state_execution import StateTaskContext

    forward, train, publish = model.forward, ModelScanInvocation.train, StateTaskContext.publish_owned_write
    fired, pending = [], []

    def cancel(owner):
        fired.append(point)
        owner.cancel_requested.set()

    def after_forward(request, layer, start, rows, *args):
        result = forward(request, layer, start, rows, *args)
        if point in ("forward", "pending") or point == "tail" and rows < chunk:
            if point == "pending":
                torch.cuda._sleep(100_000_000)
                event = torch.cuda.Event()
                event.record()
                pending.append(event)
                assert not event.query()
            cancel(context())
        return result

    def after_update(call, *args):
        result = train(call, *args)
        if point == "update":
            cancel(call.owner)
        return result

    def after_publish(call, token):
        result = publish(call, token)
        if point == "publish":
            cancel(call)
        return result

    with (
        patch.object(model, "forward", after_forward),
        patch.object(ModelScanInvocation, "train", after_update),
        patch.object(StateTaskContext, "publish_owned_write", after_publish),
    ):
        yield fired, pending
