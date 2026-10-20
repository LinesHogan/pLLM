"""Recirculation on a real vLLM model: normal and replay rows checked against an independent attention oracle.

Layer 23 is recomputed by hand (RMSNorm, rotary attention over the KV cache, SwiGLU)
for every normal and replay row; the decomposed region suffix must equal the whole
layer bitwise, cached KV may only change at the current slot, and replay must not
disturb sampling outputs or the CUDA RNG.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F
from vllm import SamplingParams
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer

import pllm
from pllm.backends.vllm.region_layers import VllmDecoderBodies
from pllm.backends.vllm.replay import VllmTokenwiseReplay
from pllm.config import ModelConfig, VllmDType
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.region_imports import BorrowedTensorImport
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_execution import RegionExecutionPlan
from pllm.execution.region_resources import RegionResources
from pllm.execution.region_storage import analyze_region_payload
from pllm.execution.replay import TokenwiseReplayTask
from pllm.execution.trace import TracePhase
from pllm.workloads.recirculation import RecirculationConfig, RecirculationWorkload


def main():
    config = RecirculationConfig(source_layer=-1, destination_layer=22, alpha=0.6, beta=None, ramp_tokens=2)
    handle = pllm.make_llm(
        workload=RecirculationWorkload(config),
        config=ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", VllmDType.BFLOAT16, 64, 2, 64, 0.25, seed=7),
    )
    # The default installs a static schedule without cold worker measurement.
    assert handle.region_selection is None
    execution = handle.deployment.execution
    layer = handle.runner.model.model.layers[23]
    native_layer, native_begin = Qwen2DecoderLayer.forward, VllmTokenwiseReplay.begin
    native_normalize, native_commit = VllmTokenwiseReplay.normalize, VllmTokenwiseReplay.commit
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
    previous_kv = {}
    normal_outputs = []
    active_row = None
    expected_suffixes = {}

    def slot_for(module):
        metadata = get_forward_context().attn_metadata[module.self_attn.attn.layer_name]
        cache = module.self_attn.attn.kv_cache
        slot = int(metadata.slot_mapping.item())
        block, offset = divmod(slot, cache.shape[2])
        return slot, cache[block, :, offset]

    def check_attention(provider, ordinal, positions, hidden, residual):
        module = provider.layers[ordinal]
        if module is layer:
            slot, current = slot_for(module)
            saved = current.clone()
            try:
                # The independent oracle checks the original whole layer. Run
                # it on private operands, then restore the current slot before
                # the actual decomposed attention consumes the same history.
                expected_suffixes[slot] = tuple(
                    value.clone()
                    for value in check_layer(
                        module, positions, hidden.clone(), None if residual is None else residual.clone()
                    )
                )
            finally:
                current.copy_(saved)
        result = native_attention(provider, ordinal, positions, hidden, residual)
        if module is layer:
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
                tensor_borrows=tuple(item for item in provider.imports if isinstance(item, BorrowedTensorImport)),
            )
            actual_suffix = suffix_execution.run(
                {"row": active_row.packed_row_index, "layer": ordinal, "hidden": result[0], "residual": result[1]},
                pages=pages,
            ).values
            expected = expected_suffixes.pop(slot)
            for actual, original in zip(actual_suffix, expected, strict=True):
                torch.testing.assert_close(actual, original, rtol=0, atol=0)
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
        cache, length = attention.attn.kv_cache, metadata.max_seq_len
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

    def check_begin(primitive, row):
        nonlocal active_row
        active_row = row
        normal_calls.append(row)
        if row.token_offset and (row.identity, row.token_offset - 1) in previous_kv:
            saved, cache, block, offset = previous_kv[(row.identity, row.token_offset - 1)]
            torch.testing.assert_close(cache[block, :, offset], saved, rtol=0, atol=0)
        return native_begin(primitive, row)

    def check_normalize(primitive, row, hidden, residual):
        output = native_normalize(primitive, row, hidden, residual)
        normal_outputs.append((output, output.clone()))
        return output

    def check_commit(primitive, row, record):
        if row.phase is HiddenPhase.PREFILL:
            replay_calls.append(row)
            metadata = primitive._tokens[row].context.attn_metadata[layer.self_attn.attn.layer_name]
            cache = layer.self_attn.attn.kv_cache
            block, offset = divmod(int(metadata.slot_mapping.item()), int(cache.shape[2]))
            previous_kv[(row.identity, row.token_offset)] = (cache[block, :, offset].clone(), cache, block, offset)
        native_commit(primitive, row, record)

    def check_task(task, context):
        rng = torch.cuda.get_rng_state()
        before = len(replay_calls)
        native_task(task, context)
        for normal, saved in normal_outputs:
            torch.testing.assert_close(normal, saved, rtol=0, atol=0)
        normal_outputs.clear()
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng, rtol=0, atol=0)
        expected = tuple(row for row in context.replay.rows if row.phase is HiddenPhase.PREFILL)
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
            patch.object(VllmTokenwiseReplay, "begin", check_begin),
            patch.object(VllmTokenwiseReplay, "normalize", check_normalize),
            patch.object(VllmTokenwiseReplay, "commit", check_commit),
            patch.object(TokenwiseReplayTask, "execute", check_task),
        ):
            outputs = handle.generate(
                ["The sky is", "Water freezes at"], SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
            )
        assert len(outputs) == 2 and all(len(output.outputs[0].token_ids) == 2 for output in outputs)
        assert len(replay_calls) == len(events) >= 4
        assert len(numerical) == len(normal_calls) + len(replay_calls)
        assert len(normal_calls) - len(replay_calls) == 2
        assert not expected_suffixes
        diagnostics = execution.diagnostics()
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        print(f"normal_rows={len(normal_calls)} replay_rows={len(replay_calls)} comparisons={len(numerical)}")
    finally:
        handle.close()


if __name__ == "__main__":
    main()
