"""Qwen2.5 BF16 oracle and a synchronization audit of each production model call."""

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
from pllm.execution.trace import TraceSelection
from test.workloads.test_model_scan import Adapters

CHUNK, RANK = 3, 2  # A real one-row tail; parameter snapshots fit the 32 KiB trace bound.


def pages(model, context, length):
    for layer in model.layers:
        attention = layer.self_attn.attn
        cache = attention.kv_cache
        positions = torch.arange(length, device=cache.device)
        blocks = context.attn_metadata[attention.layer_name].block_table[0, positions // cache.shape[2]].long()
        yield cache, (blocks, slice(None), positions % cache.shape[2], slice(None))


class Oracle:
    """Original Qwen2 forward calls and a Torch optimizer, with independent KV history."""

    def __init__(self, model, zero):
        self.model, self.zero, self.offset, self.kv = model, zero, 0, ()
        self.normal = type(model).forward.__get__(model, type(model))
        self.mlp = model.layers[0].mlp
        self.mlp_forward, self.down_forward = self.mlp.forward, self.mlp.down_proj.forward
        weight = self.mlp.down_proj.weight
        self.a = torch.empty(RANK, weight.shape[1], device=weight.device, dtype=weight.dtype)
        self.a.normal_(std=0.02, generator=torch.Generator(device=weight.device).manual_seed(7))
        if zero:
            self.a.zero_()
        self.b = torch.zeros(weight.shape[0], RANK, device=weight.device, dtype=weight.dtype)
        self.a.requires_grad_(), self.b.requires_grad_()
        self.optimizer = torch.optim.SGD((self.a, self.b), lr=1e-3, foreach=False)
        self.records = {}

    def mlp_call(self, x):
        self.target = x.clone()
        return self.mlp_forward(x)

    def down_call(self, x):
        self.x = x.clone()
        value, bias = self.down_forward(x)
        return (value if self.zero else value + (x @ self.a.T) @ self.b.T), bias

    def run(self, ids, positions):
        context, prefill = get_forward_context(), self.offset == 0
        saved = [(cache, index, cache[index].clone()) for cache, index in pages(self.model, context, self.offset)]
        for (cache, index, _), value in zip(saved, self.kv):
            cache[index] = value
        self.mlp.forward, self.mlp.down_proj.forward = self.mlp_call, self.down_call
        outputs = []
        try:
            for chunk, start in enumerate(range(0, len(positions), CHUNK)):
                stop = min(start + CHUNK, len(positions))
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
                    hidden = self.normal(ids[start:stop], positions[start:stop])
                if prefill:
                    with torch.inference_mode(False), torch.enable_grad():
                        x, target = self.x.clone(), self.target.clone()
                        prediction = x @ self.mlp.down_proj.weight.detach().T + (x @ self.a.T) @ self.b.T
                        dtype = torch.promote_types(prediction.dtype, torch.float32)
                        self.a.grad, self.b.grad = torch.autograd.grad(
                            (prediction.to(dtype) - target.to(dtype)).square().mean(), (self.a, self.b)
                        )
                    self.optimizer.step()
                self.kv = tuple(cache[index].clone() for cache, index in pages(self.model, context, length))
                record = dict(zip(("hidden", "a", "b"), (hidden.clone(), self.a.clone(), self.b.clone())))
                record.update((f"kv.{i}", value) for i, value in enumerate(self.kv))
                self.records[prefill, chunk] = record
                outputs.append(hidden)
            self.offset += len(positions)
            return torch.cat(outputs)
        finally:
            self.mlp.forward, self.mlp.down_proj.forward = self.mlp_forward, self.down_forward
            for cache, index, value in saved:
                cache[index] = value


def scenario(zero):
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
        model, oracle = causal.model, Oracle(causal.model, zero)
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
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
            tokens.append(type(causal).compute_logits(causal, reference[-1:]).argmax())
            return output

        model.forward = observed
        try:
            torch.cuda.set_sync_debug_mode("warn")  # Emit Torch's prototype notice outside the measured calls.
            torch.cuda.set_sync_debug_mode("default")
            handle.observe(TraceSelection(tensors=True))
            (result,) = handle.generate(
                ["The quick brown fox jumps over the lazy dog. Explain why."],
                SamplingParams(max_tokens=2, temperature=0, ignore_eos=True),
            )
        finally:
            model.forward = forward
        events, compared = handle.trace_snapshot().events, 0
        for event in events:
            if event.tensor_name:
                reference = oracle.records[event.row.token_offset == 0, event.iteration][event.tensor_name]
                torch.testing.assert_close(event.tensor(), reference.cpu(), atol=0, rtol=0)
                compared += 1
        diagnostics = handle.deployment.execution.diagnostics()
        updates = (len(result.prompt_token_ids) + CHUNK - 1) // CHUNK
        assert len(result.prompt_token_ids) % CHUNK and diagnostics.update_count == updates
        assert compared == updates * (len(model.layers) + 3) + len(model.layers)
        assert result.outputs[0].token_ids == [int(token) for token in tokens]
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        assert counts == dict(sync_warnings=0, event_synchronize=0, model_calls=2), counts
        return dict(zero=zero, updates=updates, compared=compared, **counts)


if __name__ == "__main__":
    # Zero-adapter original-model identity fixes atol=rtol=0 before learning is exercised.
    print(json.dumps([scenario(True), scenario(False)]))
