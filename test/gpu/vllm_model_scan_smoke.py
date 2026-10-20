"""Real five-layer BF16 oracle, executed-grid trace and per-step synchronization audit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os
import warnings
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.forward_context import get_forward_context, override_forward_context

import pllm
from pllm.execution.trace import TracePhase, TraceSelection
from test.boundary.test_region_streams import no_host_waits
from test.workloads.test_model_scan import Adapters, inject_fault, pad

CHUNK, RANK = int(os.environ.get("MODELSCAN_TEST_CHUNK", "3")), 2


def evidence(name, value):
    if directory := os.environ.get("MODELSCAN_EVIDENCE_DIR"):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(value))


def pages(model, context, length):
    for layer in model.layers:
        attention = layer.self_attn.attn
        cache = attention.kv_cache
        positions = torch.arange(length, device=cache.device)
        blocks = context.attn_metadata[attention.layer_name].block_table[0, positions // cache.shape[2]].long()
        yield cache, (blocks, slice(None), positions % cache.shape[2], slice(None))


class Oracle:
    """Original per-layer calls in chunk-major order, independent Torch optimizers and KV."""

    def __init__(self, handle, zero, span, scan, recording=True):
        self.model, self.owner, self.zero = handle.runner.get_model().model, handle.deployment.execution, zero
        self.offset, self.kv, self.records, self.params = 0, (), {}, {}
        self.prompt_length = span.prompt_length
        self.identity = span.row.identity
        self.scan = scan
        self.bindings = handle.deployment.attachment.inputs.request.overlay_bindings
        self.optimizers = {}
        self.recording = recording

    def layer(self, layer, positions, h, r):
        module = self.model.layers[layer]
        record = {"input": h.clone(), "residual_in": h.clone() if r is None else r.clone()} if self.recording else {}
        params = self.params.get(layer)
        if params is None:
            h, r = module(positions, h, r)
        else:
            a, b = params
            mlp, down = module.mlp.forward, module.mlp.down_proj.forward

            def mlp_call(x):
                record["target"] = x.clone() if self.recording else x
                return mlp(x)

            def down_call(x):
                record["x"] = x.clone() if self.recording else x
                value, bias = down(x)
                return (value if self.zero else value + (x @ a.T) @ b.T), bias

            if self.recording:
                record.update(a_read=a.clone(), b_read=b.clone())
            with (
                patch.object(module.mlp, "forward", mlp_call),
                patch.object(module.mlp.down_proj, "forward", down_call),
            ):
                h, r = module(positions, h, r)
        if self.recording:
            record.update(hidden=h.clone(), residual=r.clone())
        return h, r, record

    def run(self, ids, positions):
        context = get_forward_context()
        if self.offset == 0:
            assert self.prompt_length > 0
        prefill = self.offset < self.prompt_length
        if self.offset == 0:
            refs = tuple(
                self.scan._invocation.context.selection.ref_for(self.identity, binding.resource)
                for binding in self.bindings
            )
            with torch.inference_mode(False):
                self.params = {
                    binding.target.layer: tuple(
                        self.owner.snapshot(ref, torch.cuda.current_stream())[i].clone().requires_grad_()
                        for i in binding.fields
                    )
                    for ref, binding in zip(refs, self.bindings, strict=True)
                }
            self.optimizers = {
                layer: torch.optim.SGD(params, lr=1e-3, foreach=False) for layer, params in self.params.items()
            }
        saved = [(cache, index, cache[index].clone()) for cache, index in pages(self.model, context, self.offset)]
        for (cache, index, _), value in zip(saved, self.kv):
            cache[index] = value
        outputs = []
        try:
            chunk_size = CHUNK if prefill else 1
            for chunk, start in enumerate(range(0, len(positions), chunk_size)):
                stop = min(start + chunk_size, len(positions))
                size, length = stop - start, self.offset + stop
                metadata = {
                    name: replace(
                        base,
                        num_actual_tokens=size,
                        max_query_len=size,
                        max_seq_len=length,
                        query_start_loc=torch.tensor([0, size], dtype=torch.int32, device=positions.device),
                        seq_lens=torch.tensor([length], dtype=torch.int32, device=positions.device),
                        slot_mapping=context.slot_mapping[name][start:stop],
                        num_prefill_tokens=size if prefill else 0,
                        scheduler_metadata=None,
                        prefix_scheduler_metadata=None,
                    )
                    for name, base in context.attn_metadata.items()
                }
                with override_forward_context(
                    replace(
                        context,
                        attn_metadata=metadata,
                        slot_mapping={k: v.slot_mapping for k, v in metadata.items()},
                        batch_descriptor=None,
                        is_padding=None,
                    )
                ):
                    h, r = self.model.embed_input_ids(ids[start:stop]), None
                    for layer in range(len(self.model.layers)):
                        h, r, record = self.layer(layer, positions[start:stop], h, r)
                        if self.recording:
                            record = {
                                name: value if name.endswith("read") else pad(value, chunk_size)
                                for name, value in record.items()
                            }
                        if prefill and layer in self.params:
                            self.train(layer, record["x"][:size], record["target"][:size])
                            if self.recording:
                                a, b = self.params[layer]
                                record.update(a=a.clone(), b=b.clone())
                        if self.recording:
                            cache, index = tuple(pages(self.model, context, length))[layer]
                            record["kv"] = cache[index].clone()
                            self.records.update(
                                ((prefill, self.offset // CHUNK + chunk if prefill else 0, f"{layer}.{name}"), value)
                                for name, value in record.items()
                            )
                    outputs.append(self.model.norm(h, r)[0])
                self.kv = tuple(cache[index].clone() for cache, index in pages(self.model, context, length))
            self.offset += len(positions)
            return torch.cat(outputs)
        finally:
            for cache, index, value in saved:
                cache[index] = value

    def train(self, layer, x, target):
        a, b = self.params[layer]
        with torch.inference_mode(False), torch.enable_grad():
            x, target = x.clone(), target.clone()
            prediction = x @ self.model.layers[layer].mlp.down_proj.weight.detach().T + (x @ a.T) @ b.T
            dtype = torch.promote_types(prediction.dtype, torch.float32)
            a.grad, b.grad = torch.autograd.grad((prediction.to(dtype) - target.to(dtype)).square().mean(), (a, b))
        self.optimizers[layer].step()


def lifecycle(handle, scan):
    """Abort whole native invocations, then admit fresh requests through the same owner."""
    owner, results = handle.deployment.execution, []
    for point in ("forward", "update", "publish", "tail", "pending"):
        sequence = max((e.sequence for e in handle.trace_snapshot().events), default=-1)
        handle.observe(TraceSelection(tensors=True, max_events=4096))

        def audit(function):
            def run(*args, **kwargs):
                failure = None
                with no_host_waits(point + ":" + function.__name__):
                    try:
                        result = function(*args, **kwargs)
                    except Exception as error:
                        failure = error
                if failure is not None:
                    raise failure
                return result

            return run

        with (
            inject_fault(point, scan, lambda: scan._invocation.context, CHUNK) as (fired, pending),
            ExitStack() as stack,
        ):
            for name, method in (
                ("admit", owner.admit),
                ("start_step", owner.start_step),
                ("execute_anchor", owner.execute_anchor),
                ("cancel_step", owner.cancel_step),
                ("retire", owner.retire),
            ):
                stack.enter_context(patch.object(owner, name, audit(method)))
            try:
                handle.generate(
                    ["hello hello hello hello"], SamplingParams(max_tokens=2, temperature=0, ignore_eos=True)
                )
            except RuntimeError:
                assert fired == [point]
            else:
                raise AssertionError("fault did not cancel request")
        usage = owner.arena.usage()
        if pending:
            assert not pending[0].query()
        assert usage.private_versions == 0
        events = tuple(e for e in handle.trace_snapshot().events if e.sequence > sequence)
        publications = sum(e.phase is TracePhase.PUBLISH for e in events)
        torch.cuda.synchronize()
        owner.arena.collect_completed()
        settled = owner.arena.usage()
        assert (
            owner.arena.active_lineages
            == settled.pending_receipts
            == settled.pinned_versions
            == settled.retiring_versions
            == 0
        )
        assert publications >= 1 if point == "tail" else publications == int(point == "publish")
        results.append(
            dict(
                point=point,
                published=publications,
                pending_before=usage.pending_receipts,
                retiring_before=usage.retiring_versions,
                pending_after=settled.pending_receipts,
            )
        )
    handle.generate(["hello hello"], SamplingParams(max_tokens=1, temperature=0))
    return results


def scenario(zero, tolerance):
    counts = dict(sync_warnings=0, event_synchronize=0, model_calls=0)
    config = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=max(64, 4 * CHUNK + 4),
        max_num_seqs=3 if CHUNK == 3 else 1,
        max_num_batched_tokens=4 * CHUNK,
        gpu_memory_utilization=float(os.environ.get("MODELSCAN_TEST_MEMORY", "0.15")),
        seed=7,
    )
    with pllm.make_llm(config, workload=Adapters(chunk_size=CHUNK, rank=RANK, zero=zero)) as handle:
        from pllm.execution.codec import encode_boundary

        evidence(
            "source.json",
            json.loads(encode_boundary(handle.attachment.inputs.plan.semantic_graph.source_regions[0].program)),
        )
        causal = handle.runner.get_model()
        model, oracles = causal.model, {}
        forward, synchronize = model.forward, torch.cuda.Event.synchronize
        scan = forward.__self__
        tokens = {}
        intervals, full_records = [], []
        trace_observe = handle.deployment.execution.trace.observe

        def observe_tensor(invocation_id, task_id, name, tensor, *, row=None, iteration=None, refs=()):
            full_records.append((row, iteration, name, tensor.detach().clone(), refs))

        handle.deployment.execution.trace.observe = observe_tensor
        cell_forward = scan.forward

        def timed_cell(request, layer, start, rows, hidden, residual, parameters):
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            stream = torch.cuda.current_stream()
            begin.record(stream)
            output = cell_forward(request, layer, start, rows, hidden, residual, parameters)
            end.record(stream)
            intervals.append((scan.spans[request].row.identity, start // CHUNK, layer, stream.cuda_stream, begin, end))
            return output

        scan.forward = timed_cell
        from pllm.execution.model_scan import ModelScanInvocation

        training = ModelScanInvocation.train

        def timed_train(call, name, inputs):
            request, layer, chunk, _ = inputs[:4]
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            stream = torch.cuda.current_stream()
            begin.record(stream)
            output = training(call, name, inputs)
            end.record(stream)
            span = call.model.spans[request]
            intervals.append(
                (
                    str(span.row.identity) + ":update",
                    span.row.token_offset // CHUNK + chunk,
                    layer,
                    stream.cuda_stream,
                    begin,
                    end,
                )
            )
            return output

        def counted(event):
            counts["event_synchronize"] += 1
            return synchronize(event)

        def observed(ids, positions, intermediate_tensors=None, inputs_embeds=None):
            context, references = get_forward_context(), []
            for span in scan.spans:
                identity = span.row.identity
                oracle = oracles.setdefault(identity, Oracle(handle, zero, span, scan))
                begin, end = span.row.packed_row_index, span.row.packed_row_index + span.length
                metadata = {
                    name: replace(
                        base, block_table=base.block_table[span.row.request_ordinal : span.row.request_ordinal + 1]
                    )
                    for name, base in context.attn_metadata.items()
                }
                with override_forward_context(
                    replace(
                        context,
                        attn_metadata=metadata,
                        slot_mapping={name: value[begin:end] for name, value in context.slot_mapping.items()},
                    )
                ):
                    reference = oracle.run(ids[begin:end], positions[begin:end])
                references.append(reference)
                if oracle.offset >= oracle.prompt_length:
                    tokens.setdefault(identity, []).append(type(causal).compute_logits(causal, reference[-1:]).argmax())
            reference = torch.cat(references)
            with (
                warnings.catch_warnings(record=True) as messages,
                patch.object(torch.cuda.Event, "synchronize", counted),
            ):
                warnings.simplefilter("always")
                torch.cuda.set_sync_debug_mode("warn")
                try:
                    output = forward(ids, positions, intermediate_tensors, inputs_embeds)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            sites = [f"{w.filename}:{w.lineno}: {w.message}" for w in messages if "synchroniz" in str(w.message)]
            counts["sync_warnings"] += len(sites)
            assert not sites, sites
            counts["model_calls"] += 1
            torch.testing.assert_close(output, reference, atol=tolerance, rtol=tolerance)
            return output

        model.forward = observed
        try:
            torch.cuda.set_sync_debug_mode("warn")
            torch.cuda.set_sync_debug_mode("default")
            handle.observe(TraceSelection(tensors=True, max_events=4096))
            prompts = (
                [
                    "The quick brown fox jumps over the lazy dog. Explain why.",
                    "One two three.",
                    "What is the difference between rain and snow? Give a short answer.",
                ]
                if CHUNK == 3
                else ["hello " * (3 * CHUNK - 1) + "hello"]
            )
            with patch.object(ModelScanInvocation, "train", timed_train):
                results = handle.generate(prompts, SamplingParams(max_tokens=2, temperature=0, ignore_eos=True))
        finally:
            model.forward = forward
        handle.deployment.execution.trace.observe = trace_observe
        compared, maxima, trace = 0, {}, []
        for row, iteration, name, actual, refs in full_records:
            oracle = oracles[row.identity]
            prefill = row.token_offset < oracle.prompt_length
            reference = oracle.records[prefill, iteration, name]
            torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
            layer = int(name.split(".")[0])
            maxima[layer] = max(maxima.get(layer, 0), float((actual - reference).abs().max()))
            if prefill and name.endswith(".input"):
                trace.append((iteration, layer))
            if name.endswith("a_read"):
                assert refs[0].version_id == (iteration if prefill else (oracle.prompt_length + CHUNK - 1) // CHUNK)
            compared += 1
        diagnostics = handle.deployment.execution.diagnostics()
        chunks = sum((oracle.prompt_length + CHUNK - 1) // CHUNK for oracle in oracles.values())
        assert diagnostics.update_count == chunks * 5 and trace != sorted(trace)
        assert len(trace) == chunks * len(model.layers)
        for result, identity in zip(results, sorted(oracles, key=lambda identity: identity.prompt_idx), strict=True):
            assert result.outputs[0].token_ids == [int(token) for token in tokens[identity]]
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        assert counts == dict(sync_warnings=0, event_synchronize=0, model_calls=counts["model_calls"]), counts
        origin = intervals[0][4]
        durations = [
            (str(identity), chunk, layer, stream, origin.elapsed_time(begin), origin.elapsed_time(end))
            for identity, chunk, layer, stream, begin, end in intervals
        ]
        overlaps = [
            (a[:4], b[:4])
            for i, a in enumerate(durations)
            for b in durations[i + 1 :]
            if a[3] != b[3] and max(a[4], b[4]) < min(a[5], b[5])
        ]
        evidence("intervals-" + str(zero) + ".json", durations)
        admissions = tuple(scan.admissions)
        handle.observe(TraceSelection(tensors=False))
        intervals.clear()
        with patch.object(ModelScanInvocation, "train", timed_train):
            handle.generate(prompts, SamplingParams(max_tokens=2, temperature=0, ignore_eos=True))
        torch.cuda.synchronize()
        origin = intervals[0][4]
        warm = [
            (str(identity), chunk, layer, stream, origin.elapsed_time(begin), origin.elapsed_time(end))
            for identity, chunk, layer, stream, begin, end in intervals
        ]
        warm_overlaps = [
            (a[:4], b[:4])
            for i, a in enumerate(warm)
            for b in warm[i + 1 :]
            if a[3] != b[3] and max(a[4], b[4]) < min(a[5], b[5])
        ]
        evidence("warm-intervals-" + str(zero) + ".json", warm)
        faults = lifecycle(handle, scan) if not zero and CHUNK == 3 else []
        return dict(
            faults=faults,
            warm_overlaps=len(warm_overlaps),
            warm_overlap_examples=warm_overlaps[:4],
            chunk_size=CHUNK,
            identities=[str(identity) for identity in oracles],
            overlap_examples=overlaps[:4],
            overlaps=len(overlaps),
            admissions=[
                dict(tokens=[g.tokens for g in batch.grants], unused=batch.unused_tokens) for batch in admissions
            ],
            zero=zero,
            updates=chunks * 5,
            compared=compared,
            maximum_errors=maxima,
            trace=trace,
            **counts,
        )


if __name__ == "__main__":
    baseline = scenario(True, 0)
    # The no-update run fixes zero tolerance before the learned trajectory runs.
    learned = scenario(False, 0)
    assert set(baseline["identities"]).isdisjoint(learned["identities"])
    print(json.dumps([baseline, learned]))
