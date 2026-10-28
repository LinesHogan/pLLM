"""Batched recirculation checked against independent eager attention and feed-forward math."""

import json
import sys
from dataclasses import replace
from functools import partial
from math import gcd
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.forward_context import get_forward_context, override_forward_context

import pllm
from pllm.backends.vllm import runtime
from pllm.backends.vllm.replay import RouteWave, VllmTokenwiseReplay
from pllm.config import ModelConfig, VllmDType
from pllm.workloads.recirculation import JacobiRecirculation, RecirculationWorkload


def replay_handle(
    workload,
    *,
    length=64,
    prefix=False,
    blocks=None,
    offload=None,
    generated_kv="isolate",
    multiprocessing=False,
    batch=2,
    budget=None,
    chunked=False,
):
    """Shared small-model Stage A engine setup for serial-route GPU evidence."""
    with patch.object(
        runtime, "LLM", runtime.LLM if blocks is None else partial(runtime.LLM, num_gpu_blocks_override=blocks)
    ):
        return pllm.make_llm(
            workload=workload,
            config=ModelConfig(
                "Qwen/Qwen2.5-0.5B-Instruct",
                VllmDType.BFLOAT16,
                length,
                batch,
                length if budget is None else budget,
                0.25,
                seed=7,
                enable_prefix_caching=prefix,
                enable_chunked_prefill=chunked,
                kv_offload=offload,
                generated_kv=generated_kv,
                multiprocessing=multiprocessing,
            ),
        )


def rms(value, norm):
    return (
        value.float()
        * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + norm.variance_epsilon)
        * norm.weight.float()
    ).to(norm.weight.dtype)


def layer_reference(layer, positions, hidden, residual, prior):
    """Independent eager layer math shared by serial and Jacobi convergence evidence."""
    attention = layer.self_attn
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
    rounding_bound = 2 * torch.finfo(hidden.dtype).eps * (expected_residual.float().abs() + value.float().abs()) + 0.02
    value = rms(residual_float, layer.post_attention_layernorm)
    gate, up = F.linear(value, layer.mlp.gate_up_proj.weight, layer.mlp.gate_up_proj.bias).chunk(2, -1)
    value = (F.silu(gate.float()) * up.float()).to(hidden.dtype)
    expected = F.linear(value, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias)
    return expected, residual_float, expected_kv, rounding_bound


def main():
    chunked = "--chunked" in sys.argv
    handle = replay_handle(
        RecirculationWorkload(destination=22, alpha=0.6, ramp_tokens=2),
        chunked=chunked,
        budget=7 if chunked else None,
        prefix="--prefix" in sys.argv,
    )
    primitive = handle.runner.require_session()._replay
    layer = primitive.model.layers[23]
    native_layer, native_capture, native_replay = RouteWave.layer, RouteWave.capture, torch.cuda.CUDAGraph.replay
    native_forward = VllmTokenwiseReplay.forward
    recordings, numerical, checked = {}, [], []
    active_records = None

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

    def capture(wave):
        nonlocal active_records
        active_records = []
        cells = [(a.kv_cache, wave.attention.metadata[a.layer_name].slot_mapping) for a in primitive.attentions]
        before = [cache[slots // cache.shape[2], :, slots % cache.shape[2]].clone() for cache, slots in cells]
        native_capture(wave)
        for (cache, slots), saved in zip(cells, before):
            torch.testing.assert_close(
                cache[slots // cache.shape[2], :, slots % cache.shape[2]], saved, rtol=0, atol=0, equal_nan=True
            )
        recordings[wave.graph] = wave, active_records
        active_records = None

    def called(wave, ordinal, hidden, residual, kv="commit"):
        selected = ordinal == 23 and torch.cuda.is_current_stream_capturing()
        operands = (
            (wave.positions.clone(), hidden.clone(), None if residual is None else residual.clone()) if selected else ()
        )
        result = native_layer(wave, ordinal, hidden, residual, kv)
        if selected:
            cache = layer.self_attn.attn.kv_cache
            slots = wave.attention.metadata[layer.self_attn.attn.layer_name].slot_mapping
            current = cache[slots // cache.shape[2], :, slots % cache.shape[2]].clone()
            active_records.append((operands, tuple(value.clone() for value in result), current))
        return result

    def check_layer(positions, hidden, residual, actual, actual_residual, observed_kv, metadata, lane):
        attention = layer.self_attn
        cache, length = attention.attn.kv_cache, int(metadata.seq_lens[lane])
        block_table, block_size = metadata.block_table[lane], cache.shape[2]
        prior = torch.stack(
            [cache[int(block_table[p // block_size]), :, p % block_size].clone() for p in range(length)]
        )
        expected, residual_float, expected_kv, rounding_bound = layer_reference(
            layer, positions, hidden, residual, prior
        )
        residual_error = (actual_residual.float() - residual_float.to(hidden.dtype).float()).abs()
        assert torch.all(residual_error <= rounding_bound), (residual_error.max().item(), rounding_bound.max().item())
        error = (actual.float() - expected.float()).abs()
        assert error.square().sum().sqrt() / expected.float().square().sum().sqrt() < 0.015
        block, offset = divmod(int(metadata.slot_mapping[lane]), int(block_size))
        torch.testing.assert_close(observed_kv, expected_kv, rtol=0.02, atol=0.02)
        for p in range(length - 1):
            torch.testing.assert_close(
                cache[int(block_table[p // block_size]), :, p % block_size], prior[p], rtol=0, atol=0
            )
        numerical.append(lane)
        return actual, actual_residual

    def replay(graph):
        native_replay(graph)
        if graph not in recordings:
            return
        wave, records = recordings[graph]
        metadata = wave.attention.metadata[layer.self_attn.attn.layer_name]
        for operands, outputs, kv in records:
            for lane in range(wave.seed.shape[0]):
                check_layer(
                    operands[0][lane : lane + 1],
                    operands[1][lane : lane + 1],
                    None if operands[2] is None else operands[2][lane : lane + 1],
                    outputs[0][lane : lane + 1],
                    outputs[1][lane : lane + 1],
                    kv[lane],
                    metadata,
                    lane,
                )
        normal = primitive.model.norm(records[0][1][0] + records[0][1][1])
        torch.testing.assert_close(wave.output, normal, rtol=0, atol=0)
        checked.append(wave.seed.shape[0])

    try:
        with (
            patch.object(RouteWave, "layer", called),
            patch.object(RouteWave, "capture", capture),
            patch.object(torch.cuda.CUDAGraph, "replay", replay),
            patch.object(primitive.model, "forward", partial(sparse_forward, primitive)),
        ):
            tokens = []
            for iteration in range(4):
                if iteration == 3:
                    layer.self_attn.attn.kv_cache = layer.self_attn.attn.kv_cache.clone()
                outputs = handle.generate(
                    ["The sky is " * 7, "Water freezes at " * 7],
                    SamplingParams(max_tokens=2, temperature=0, ignore_eos=True),
                )
                tokens.append([list(output.outputs[0].token_ids) for output in outputs])
        assert all(value == tokens[0] for value in tokens)
        assert 2 in checked and len(numerical) == 2 * sum(checked)
        diagnostics = handle.diagnostics()
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        print(f"recirculation waves={len(checked)} independent layer comparisons={len(numerical)}")
    finally:
        handle.close()


@torch.no_grad()
def serial_reference(model, ids):
    """Run both passes position by position, with a separate cache and greedy continuation."""
    cache = [
        model.model.norm.weight.new_zeros((len(ids) + 31, layer.self_attn.num_kv_heads, 2 * layer.self_attn.head_dim))
        for layer in model.model.layers
    ]
    hidden_rows, generated = [], []
    for position in range(len(ids) + 31):
        hidden = model.model.embed_input_ids(
            torch.tensor([ids[position] if position < len(ids) else generated[-1]], device="cuda")
        )
        residual, taps = None, []
        for ordinal in (*range(len(cache)), *range(1, len(cache))):
            if len(taps) == len(cache) and ordinal == 1:
                source, destination = taps[-1].float(), taps[0].float()
                hidden, residual = (
                    (
                        0.85 * destination
                        + 0.15
                        * source
                        * destination.norm(dim=-1, keepdim=True)
                        / source.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
                    ).to(hidden.dtype),
                    None,
                )
            hidden, carry, kv, _ = layer_reference(
                model.model.layers[ordinal],
                torch.tensor([position], device="cuda"),
                hidden,
                residual,
                cache[ordinal][: position + 1],
            )
            residual = carry.to(hidden.dtype)
            cache[ordinal][position] = kv
            if len(taps) < len(cache):
                taps.append(hidden + residual)
        normal = model.model.norm(taps[-1])
        hidden_rows.append(normal)
        if position >= len(ids) - 1:
            generated.append(int(F.linear(normal, model.lm_head.weight).argmax(-1)))
    return torch.cat(hidden_rows[: len(ids)]).cpu(), generated


@torch.no_grad()
def jacobi_evidence():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    prompt = tokenizer.decode(
        tokenizer.encode(
            "Janet has 16 apples. She eats three and gives four to her friends. She sells the rest for two dollars each. How much money does she make? "
            * 12
        )[:256]
    )
    ids = tokenizer.encode(prompt)
    assert len(ids) == 256
    for rounds in (0, 1, 2, 3):
        handle = replay_handle(
            RecirculationWorkload() if rounds == 0 else JacobiRecirculation(rounds=rounds),
            length=320,
            batch=8,
            budget=2048,
        )
        primitive = handle.runner.require_session()._replay
        native, records = primitive.model.forward, []

        def forward(*args, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = native(*args, **kwargs)
            end.record()
            if result.shape[0] == 2048:
                records.append((start, end, result[:256].clone()))
            return result

        try:
            if rounds == 0:
                reference, tokens = serial_reference(handle.runner.model, ids)
            primitive.model.forward = forward
            for _ in range(2):
                outputs = handle.generate([prompt] * 8, SamplingParams(max_tokens=32, temperature=0, ignore_eos=True))
            torch.cuda.synchronize()
            start, end, actual = records[-1]
            error = (actual.float() - reference.cuda().float()).norm(dim=-1) / reference.cuda().float().norm(dim=-1)
            expected_logp, actual_logp = (
                F.linear(h[-1:], handle.runner.model.lm_head.weight).float().log_softmax(-1)
                for h in (reference.cuda(), actual)
            )
            agreement = sum(a == b for a, b in zip(tokens, outputs[0].outputs[0].token_ids, strict=True))
            print(
                "JACOBI "
                + json.dumps(
                    dict(
                        rounds=rounds,
                        hidden_max=float(error.max()),
                        hidden_mean=float(error.mean()),
                        kl=float((expected_logp.exp() * (expected_logp - actual_logp)).sum()),
                        greedy=agreement,
                        prefill_ms=start.elapsed_time(end),
                    )
                ),
                flush=True,
            )
            assert torch.isfinite(error).all() and agreement == 32 and (rounds < 2 or float(error.mean()) < 0.04)
        finally:
            handle.close()


if __name__ == "__main__":
    jacobi_evidence() if "--jacobi" in sys.argv else main()
