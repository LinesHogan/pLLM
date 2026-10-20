"""Real five-layer BF16 oracle, executed-grid trace and per-step synchronization audit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import warnings
from dataclasses import replace
from unittest.mock import patch

import torch
from vllm import SamplingParams
from vllm.forward_context import get_forward_context, override_forward_context

import pllm
from pllm.execution.trace import TracePhase, TraceSelection
from test.workloads.test_model_scan import Adapters, pad

CHUNK, RANK = 3, 2


def pages(model, context, length):
    for layer in model.layers:
        attention = layer.self_attn.attn
        cache = attention.kv_cache
        positions = torch.arange(length, device=cache.device)
        blocks = context.attn_metadata[attention.layer_name].block_table[0, positions // cache.shape[2]].long()
        yield cache, (blocks, slice(None), positions % cache.shape[2], slice(None))


class Oracle:
    """Original per-layer calls in chunk-major order, independent Torch optimizers and KV."""

    def __init__(self, handle, zero):
        self.model, self.owner, self.zero = handle.runner.get_model().model, handle.deployment.execution, zero
        self.offset, self.kv, self.records, self.params = 0, (), {}, {}
        self.bindings = handle.deployment.attachment.inputs.request.overlay_bindings
        self.optimizers = {}

    def layer(self, layer, positions, h, r):
        module = self.model.layers[layer]
        record = {"input": h.clone(), "residual_in": h.clone() if r is None else r.clone()}
        params = self.params.get(layer)
        if params is None:
            h, r = module(positions, h, r)
        else:
            a, b = params
            mlp, down = module.mlp.forward, module.mlp.down_proj.forward

            def mlp_call(x):
                record["target"] = x.clone()
                return mlp(x)

            def down_call(x):
                record["x"] = x.clone()
                value, bias = down(x)
                return (value if self.zero else value + (x @ a.T) @ b.T), bias

            record.update(a_read=a.clone(), b_read=b.clone())
            with (
                patch.object(module.mlp, "forward", mlp_call),
                patch.object(module.mlp.down_proj, "forward", down_call),
            ):
                h, r = module(positions, h, r)
        record.update(hidden=h.clone(), residual=r.clone())
        return h, r, record

    def run(self, ids, positions):
        context, prefill = get_forward_context(), self.offset == 0
        if prefill:
            refs = next(e.refs for e in reversed(self.owner.trace.snapshot().events) if e.phase is TracePhase.ADMIT)
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
                        record = {
                            name: value if name.endswith("read") else pad(value, chunk_size)
                            for name, value in record.items()
                        }
                        if prefill and layer in self.params:
                            a, b = self.params[layer]
                            with torch.inference_mode(False), torch.enable_grad():
                                x, target = record["x"][:size].clone(), record["target"][:size].clone()
                                prediction = (
                                    x @ self.model.layers[layer].mlp.down_proj.weight.detach().T + (x @ a.T) @ b.T
                                )
                                dtype = torch.promote_types(prediction.dtype, torch.float32)
                                a.grad, b.grad = torch.autograd.grad(
                                    (prediction.to(dtype) - target.to(dtype)).square().mean(), (a, b)
                                )
                            self.optimizers[layer].step()
                            record.update(a=a.clone(), b=b.clone())
                        cache, index = tuple(pages(self.model, context, length))[layer]
                        record["kv"] = cache[index].clone()
                        self.records.update(
                            ((prefill, chunk, f"{layer}.{name}"), value) for name, value in record.items()
                        )
                    outputs.append(self.model.norm(h, r)[0])
                self.kv = tuple(cache[index].clone() for cache, index in pages(self.model, context, length))
            self.offset += len(positions)
            return torch.cat(outputs)
        finally:
            for cache, index, value in saved:
                cache[index] = value


def scenario(zero, tolerance):
    counts = dict(sync_warnings=0, event_synchronize=0, model_calls=0)
    config = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.15,
        seed=7,
    )
    with pllm.make_llm(config, workload=Adapters(chunk_size=CHUNK, rank=RANK, zero=zero)) as handle:
        causal = handle.runner.get_model()
        model, oracle = causal.model, Oracle(handle, zero)
        forward, synchronize = model.forward, torch.cuda.Event.synchronize
        tokens = []

        def counted(event):
            counts["event_synchronize"] += 1
            return synchronize(event)

        def observed(ids, positions, intermediate_tensors=None, inputs_embeds=None):
            reference = oracle.run(ids, positions)
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
            tokens.append(type(causal).compute_logits(causal, reference[-1:]).argmax())
            return output

        model.forward = observed
        try:
            torch.cuda.set_sync_debug_mode("warn")
            torch.cuda.set_sync_debug_mode("default")
            handle.observe(TraceSelection(tensors=True, max_events=4096))
            (result,) = handle.generate(
                ["The quick brown fox jumps over the lazy dog. Explain why."],
                SamplingParams(max_tokens=2, temperature=0, ignore_eos=True),
            )
        finally:
            model.forward = forward
        events, compared, maxima, trace = handle.trace_snapshot().events, 0, {}, []
        for event in events:
            if event.tensor_name:
                reference = oracle.records[event.row.token_offset == 0, event.iteration, event.tensor_name].cpu()
                actual = event.tensor()
                torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
                layer = int(event.tensor_name.split(".")[0])
                maxima[layer] = max(maxima.get(layer, 0), float((actual - reference).abs().max()))
                if event.row.token_offset == 0 and event.tensor_name.endswith(".input"):
                    trace.append((event.iteration, layer))
                if event.tensor_name.endswith("a_read"):
                    assert event.refs[0].version_id == (
                        event.iteration
                        if event.row.token_offset == 0
                        else (len(result.prompt_token_ids) + CHUNK - 1) // CHUNK
                    )
                compared += 1
        diagnostics = handle.deployment.execution.diagnostics()
        chunks = (len(result.prompt_token_ids) + CHUNK - 1) // CHUNK
        assert diagnostics.update_count == chunks * 5 and trace != sorted(trace)
        assert len(trace) == chunks * len(model.layers)
        assert result.outputs[0].token_ids == [int(token) for token in tokens]
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        assert counts == dict(sync_warnings=0, event_synchronize=0, model_calls=2), counts
        return dict(zero=zero, updates=chunks * 5, compared=compared, maximum_errors=maxima, trace=trace, **counts)


if __name__ == "__main__":
    baseline = scenario(True, 0)
    # The no-update run fixes zero tolerance before the learned trajectory runs.
    print(json.dumps([baseline, scenario(False, 0)]))
