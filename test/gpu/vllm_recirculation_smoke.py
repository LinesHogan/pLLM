"""Recirculation on a real vLLM model: normal and replay rows checked against an independent attention oracle.

Layer 23 is recomputed by hand (RMSNorm, rotary attention over the KV cache, SwiGLU)
for every normal and replay row; the decomposed region suffix must equal the whole
layer bitwise, cached KV may only change at the current slot, and replay must not
disturb sampling outputs or the CUDA RNG.
"""

import sys
from dataclasses import replace
from math import gcd
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F
from vllm import SamplingParams
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

import pllm
from pllm.backends.vllm.region_attention import TokenAttentionBuffers
from pllm.backends.vllm.region_layers import VllmDecoderBodies
from pllm.backends.vllm.replay import VllmTokenwiseReplay
from pllm.config import ModelConfig, VllmDType
from pllm.contracts.region_imports import BorrowedTensorImport
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_execution import RegionExecutionPlan, RegionTokenCapture
from pllm.execution.region_resources import RegionResources
from pllm.execution.region_storage import analyze_region_payload
from pllm.execution.replay import TokenwiseReplayTask
from pllm.execution.trace import TracePhase
from pllm.workloads.recirculation import RecirculationWorkload


def main():
    handle = pllm.make_llm(
        workload=RecirculationWorkload(destination=22, alpha=0.6, ramp_tokens=2),
        config=ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", VllmDType.BFLOAT16, 64, 2, 64, 0.25, seed=7),
    )
    execution = handle.deployment.execution
    layer = handle.runner.model.model.layers[23]
    native_layer = Qwen2DecoderLayer.forward
    native_normalize = VllmTokenwiseReplay.normalize
    native_forward, native_token_capture = VllmTokenwiseReplay.forward, VllmTokenwiseReplay._capture_token
    native_capture, native_refresh = RegionTokenCapture.__init__, TokenAttentionBuffers.refresh
    native_replay = torch.cuda.CUDAGraph.replay
    native_task = TokenwiseReplayTask.execute
    native_attention = VllmDecoderBodies.attention
    primitive = handle.runner._workload_session._replay
    provider = primitive._layer_bodies
    suffix = provider.suffix.program
    schedule = analyze_region_program(suffix, optimize=False).reference_schedule
    proposal = RegionScheduleProposal(analyze_region_payload(suffix, schedule, provider.imports))
    suffix_execution = RegionExecutionPlan(suffix, provider.imports, proposal, proposal.peak_bytes).bind(
        provider.functions
    )
    events, normal_calls, replay_calls, numerical = [], [], [], []
    previous_kv, recordings = {}, {}
    normal_outputs, retained_outputs = [], []
    active_row, active_records = None, None

    def sparse_forward(primitive, *args, **kwargs):
        context = get_forward_context()
        metadata, mappings = {}, {}
        for ordinal, module in enumerate(primitive.model.layers):
            attention = module.self_attn.attn
            item = context.attn_metadata[attention.layer_name]
            blocks, block_size = attention.kv_cache.shape[0], attention.kv_cache.shape[2]
            stride = next(value for value in range(2, blocks) if gcd(value, blocks) == 1)
            slots = context.slot_mapping[attention.layer_name]
            slots = ((slots // block_size * stride + ordinal) % blocks) * block_size + slots % block_size
            metadata[attention.layer_name] = replace(
                item, block_table=(item.block_table * stride + ordinal) % blocks, slot_mapping=slots
            )
            mappings[attention.layer_name] = slots
        with override_forward_context(replace(context, attn_metadata=metadata, slot_mapping=mappings)):
            return native_forward(primitive, *args, **kwargs)

    def check_token_capture(primitive, invocation, row, signature):
        slots = tuple(
            (module.self_attn.attn.kv_cache, *map(int, primitive._access.cell(module.self_attn.attn.layer_name, row)))
            for module in primitive.model.layers
        )
        saved = tuple(cache[block, :, offset].clone() for cache, block, offset in slots)
        capture = native_token_capture(primitive, invocation, row, signature)
        for (cache, block, offset), value in zip(slots, saved, strict=True):
            torch.testing.assert_close(cache[block, :, offset], value, rtol=0, atol=0)
        return capture

    def check_capture(capture, invocation, instances, device):
        nonlocal active_records
        active_records = []
        native_capture(capture, invocation, instances, device)
        recordings[capture.graph] = active_records
        active_records = None

    def check_refresh(bindings, row):
        nonlocal active_row
        active_row = row
        native_refresh(bindings, row)

    def check_attention(provider, ordinal, positions, hidden, residual):
        selected = ordinal == 23 and torch.cuda.is_current_stream_capturing()
        operands = (
            (positions.clone(), hidden.clone(), None if residual is None else residual.clone()) if selected else ()
        )
        result = native_attention(provider, ordinal, positions, hidden, residual)
        if selected:
            metadata = get_forward_context().attn_metadata[layer.self_attn.attn.layer_name]
            cache = layer.self_attn.attn.kv_cache
            slot = metadata.slot_mapping
            current = cache[slot // cache.shape[2], :, slot % cache.shape[2], :].clone().squeeze(0)
            active_records.append((operands, tuple(value.clone() for value in result), current))
        return result

    def rms(value, norm):
        return (
            value.float()
            * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + norm.variance_epsilon)
            * norm.weight.float()
        ).to(norm.weight.dtype)

    def check_layer(module, positions, hidden, residual):
        if module is not layer:
            return native_layer(module, positions, hidden, residual)
        attention = layer.self_attn
        metadata = get_forward_context().attn_metadata[attention.attn.layer_name]
        cache, length = attention.attn.kv_cache, int(metadata.seq_lens.item())
        block_table, block_size = metadata.block_table[0], cache.shape[2]
        prior = torch.stack(
            [cache[int(block_table[p // block_size]), :, p % block_size].clone() for p in range(length)]
        )
        carry = hidden.float() if residual is None else hidden.float() + residual.float()
        expected_residual = carry.to(hidden.dtype)
        value = rms(carry, layer.input_layernorm)
        qkv = F.linear(value, attention.qkv_proj.weight, attention.qkv_proj.bias)
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], -1)
        q, k, v = (part.reshape(-1, attention.head_dim) for part in (q, k, v))
        cos, sin = attention.rotary_emb.cos_sin_cache[int(positions.item())].chunk(2)

        def rotate(value):
            first, second = value.chunk(2, -1)
            return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(hidden.dtype)

        q, k = rotate(q), rotate(k)
        expected_kv = torch.cat((k, v), -1)
        history = prior.clone()
        history[-1] = expected_kv
        keys, values = history.chunk(2, -1)
        repeats = attention.num_heads // attention.num_kv_heads
        keys, values = (part.repeat_interleave(repeats, 1).transpose(0, 1).float() for part in (keys, values))
        scores = (q.float().unsqueeze(1) * keys).sum(-1) * attention.scaling
        attended = (scores.softmax(-1).unsqueeze(-1) * values).sum(1).reshape(1, -1).to(hidden.dtype)
        value = F.linear(attended, attention.o_proj.weight, attention.o_proj.bias)
        residual_float = expected_residual.float() + value.float()
        rounding_bound = (
            2 * torch.finfo(hidden.dtype).eps * (expected_residual.float().abs() + value.float().abs()) + 0.02
        )
        value = rms(residual_float, layer.post_attention_layernorm)
        gate, up = F.linear(value, layer.mlp.gate_up_proj.weight, layer.mlp.gate_up_proj.bias).chunk(2, -1)
        value = (F.silu(gate.float()) * up.float()).to(hidden.dtype)
        expected = F.linear(value, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias)
        actual, actual_residual = native_layer(module, positions, hidden, residual)
        residual_error = (actual_residual.float() - residual_float.to(hidden.dtype).float()).abs()
        assert torch.all(residual_error <= rounding_bound), (residual_error.max().item(), rounding_bound.max().item())
        error = (actual.float() - expected.float()).abs()
        assert error.square().sum().sqrt() / expected.float().square().sum().sqrt() < 0.015
        block, offset = divmod(int(metadata.slot_mapping.item()), int(block_size))
        torch.testing.assert_close(cache[block, :, offset], expected_kv, rtol=0.02, atol=0.02)
        for p in range(length - 1):
            torch.testing.assert_close(
                cache[int(block_table[p // block_size]), :, p % block_size], prior[p], rtol=0, atol=0
            )
        numerical.append(active_row)
        return actual, actual_residual

    def check_replay(graph):
        native_replay(graph)
        if graph not in recordings:
            return
        capture = next(item for item in primitive._captures.values() if item.graph is graph)
        row = active_row
        normal_calls.append(row)
        cache = layer.self_attn.attn.kv_cache
        with override_forward_context(primitive._buffers.context):
            metadata = get_forward_context().attn_metadata[layer.self_attn.attn.layer_name]
            block, offset = divmod(int(metadata.slot_mapping.item()), cache.shape[2])
            for (identity, position), saved in previous_kv.items():
                if identity == row.identity and position < row.token_offset:
                    value, prior_block, prior_offset = saved
                    torch.testing.assert_close(cache[prior_block, :, prior_offset], value, rtol=0, atol=0)
            live = cache[block, :, offset].clone()
            try:
                for index, (operands, attention_outputs, observed_kv) in enumerate(recordings[graph]):
                    expected = check_layer(layer, *operands)
                    torch.testing.assert_close(observed_kv, cache[block, :, offset], rtol=0, atol=0)
                    owner = primitive._invocation.context
                    context_resource, weight_resource = suffix.resources
                    pages = RegionResources(
                        suffix,
                        owner,
                        (),
                        borrows=(
                            primitive.bind_context(context_resource, owner),
                            primitive.bind_weights(weight_resource, owner),
                        ),
                        tensor_borrows=tuple(
                            item for item in provider.imports if isinstance(item, BorrowedTensorImport)
                        ),
                    )
                    actual = suffix_execution.run(
                        {
                            "row": row.packed_row_index,
                            "layer": 23,
                            "hidden": attention_outputs[0],
                            "residual": attention_outputs[1],
                        },
                        pages=pages,
                    ).values
                    for value, whole in zip(actual, expected, strict=True):
                        torch.testing.assert_close(value, whole, rtol=0, atol=0)
                    if index == 0:
                        torch.testing.assert_close(
                            capture.output,
                            native_normalize(primitive, row, actual[0] + actual[1], torch.zeros_like(actual[0])),
                            rtol=0,
                            atol=0,
                        )
                        output = native_normalize(primitive, row, *actual)
                        # The materialized bf16 stream rounds before final norm; main fused the addition.
                        torch.testing.assert_close(capture.output, output, rtol=0.02, atol=0.02)
                        normal_outputs.append(capture.output.clone())
            finally:
                cache[block, :, offset].copy_(live)
            replay_calls.append(row)
            previous_kv[row.identity, row.token_offset] = (live, block, offset)

    def check_task(task, context):
        rng = torch.cuda.get_rng_state()
        before = len(replay_calls)
        native_task(task, context)
        retained_outputs.append((primitive._output, torch.cat(normal_outputs)))
        normal_outputs.clear()
        for output, saved in retained_outputs:
            torch.testing.assert_close(output, saved, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng, rtol=0, atol=0)
        expected = context.replay.rows
        assert tuple(replay_calls[before:]) == expected
        assert context.publication is None and tuple(context.pin.refs) == ()
        events.extend(
            event
            for event in context.trace.snapshot().events
            if event.invocation_id == context.invocation_id and event.phase is TracePhase.REPLAY
        )

    try:
        with (
            patch.object(VllmDecoderBodies, "attention", check_attention),
            patch.object(RegionTokenCapture, "__init__", check_capture),
            patch.object(TokenAttentionBuffers, "refresh", check_refresh),
            patch.object(torch.cuda.CUDAGraph, "replay", check_replay),
            patch.object(VllmTokenwiseReplay, "forward", sparse_forward),
            patch.object(VllmTokenwiseReplay, "_capture_token", check_token_capture),
            patch.object(TokenwiseReplayTask, "execute", check_task),
        ):
            for iteration in range(4):
                if iteration == 3:
                    layer.self_attn.attn.kv_cache = layer.self_attn.attn.kv_cache.clone()
                outputs = handle.generate(
                    ["The sky is " * 7, "Water freezes at " * 7],
                    SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True),
                )
        assert len(outputs) == 2 and all(len(output.outputs[0].token_ids) == 2 for output in outputs)
        assert len(replay_calls) == len(events) >= 4
        assert len(numerical) == len(normal_calls) + len(replay_calls)
        assert len(normal_calls) == len(replay_calls)
        assert len(recordings) == 2
        diagnostics = execution.diagnostics()
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        print(f"normal_rows={len(normal_calls)} replay_rows={len(replay_calls)} comparisons={len(numerical)}")
    finally:
        handle.close()


if __name__ == "__main__":
    main()
