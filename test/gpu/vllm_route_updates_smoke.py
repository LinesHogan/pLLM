"""Fixed route re-runs train private overlay fields and match eager Qwen/KV oracles under Stage A."""

from dataclasses import dataclass, replace
from unittest.mock import patch

import torch
from torch.nn import functional as F
from vllm import SamplingParams
from vllm.forward_context import override_forward_context

import pllm
from pllm.config import ModelConfig
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload


@dataclass(frozen=True)
class SettledInPlaceTTT(InPlaceTTTWorkload):
    """Apply first-pass operands, then re-run the final two layers a fixed number of times."""

    iterations: int = 2

    @pllm.layer_route(tokens="decode")
    def settle(self, layers: pllm.Layers) -> torch.Tensor:
        taps = layers()
        layers.apply_updates()
        for _ in range(self.iterations):
            layers(start=22, stop=24, hidden=taps[21])
        return taps[-1]


def run(chunk):
    targets = tuple(
        ParameterTarget(layer, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT) for layer in (22, 23)
    )
    workload = SettledInPlaceTTT(InPlaceTTTConfig(targets, chunk_size=chunk, learning_rate=1e-6, update_decode=True))
    handle = pllm.make_llm(
        workload=workload,
        config=ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", "bfloat16", 64, 2, 64, 0.25, seed=7, cuda_graphs=False),
    )
    primitive = handle.runner.require_session()._replay
    native, checked, errors = torch.cuda.CUDAGraph.replay, [], []

    def rms(value, norm):
        return (
            value.float()
            * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + norm.variance_epsilon)
            * norm.weight.float()
        ).to(norm.weight.dtype)

    def oracle_layer(layer, parts, context, position, delta, qkv_input):
        """Independent per-layer replay math; layers chain as (hidden, residual) pairs.

        The fused add-RMSNorm norms the unrounded fp32 residual sum and rounds
        only the stored residual — the oracle must hand over the pair, not a
        pre-rounded bf16 sum."""
        hidden_in, residual_in = parts
        if residual_in is None:
            residual = hidden_in
            hidden = rms(hidden_in, layer.input_layernorm)
        else:
            summed = hidden_in.float() + residual_in.float()
            residual = summed.to(hidden_in.dtype)
            hidden = rms(summed, layer.input_layernorm)
        attention = layer.self_attn
        cache = attention.attn.kv_cache
        metadata = context.attn_metadata[attention.attn.layer_name]
        block_table = metadata.block_table[0]
        length, block_size = int(metadata.seq_lens[0].item()), cache.shape[2]
        # Copy only valid history, not the model's large mostly unused KV allocation.
        history = torch.stack(
            [cache[int(block_table[pos // block_size]), :, pos % block_size] for pos in range(length)]
        )
        qkv = F.linear(hidden, attention.qkv_proj.weight, attention.qkv_proj.bias)
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], -1)
        q, k, v = (value.reshape(-1, attention.head_dim) for value in (q, k, v))
        cos, sin = attention.rotary_emb.cos_sin_cache[int(position.item())].chunk(2)

        def rotate(value):
            first, second = value.float().chunk(2, -1)
            return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(hidden_in.dtype)

        q, k = rotate(q), rotate(k)
        expected_kv = torch.cat((k, v), -1)
        history[-1] = expected_kv
        keys, values = history.chunk(2, -1)
        repeat = attention.num_heads // attention.num_kv_heads
        keys, values = (value.repeat_interleave(repeat, dim=1).transpose(0, 1).float() for value in (keys, values))
        scores = (q.float().unsqueeze(1) * keys).sum(-1) * attention.scaling
        attended = (scores.softmax(-1).unsqueeze(-1) * values).sum(1).reshape(1, -1).to(hidden_in.dtype)
        hidden = F.linear(attended, attention.o_proj.weight, attention.o_proj.bias)
        residual_fp32 = residual.float() + hidden.float()
        residual = residual_fp32.to(hidden_in.dtype)
        hidden = rms(residual_fp32, layer.post_attention_layernorm)
        gate, up = F.linear(hidden, layer.mlp.gate_up_proj.weight, layer.mlp.gate_up_proj.bias).chunk(2, -1)
        hidden = (F.silu(gate.float()) * up.float()).to(hidden_in.dtype)
        expected = F.linear(hidden, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias)
        expected += F.linear(hidden, delta)
        # Isolate KV projection/RoPE from the RMSNorm rounding checked by the full-layer oracle.
        _, reference_key, reference_value = F.linear(
            qkv_input, attention.qkv_proj.weight, attention.qkv_proj.bias
        ).split([attention.q_size, attention.kv_size, attention.kv_size], -1)
        return (
            expected,
            residual,
            torch.cat(
                (
                    rotate(reference_key.reshape(-1, attention.head_dim)),
                    reference_value.reshape(-1, attention.head_dim),
                ),
                -1,
            ),
        )

    def replay(graph):
        capture = next(value for value in primitive.captures.values() if value.graph is graph)
        ports = tuple(
            value for adapter in capture.routes for value in (*adapter.entry_fields, *adapter.fields, *adapter.saved)
        )
        fields = tuple(value.clone() for value in ports)
        slots = tuple(
            (layer.self_attn.attn.kv_cache, capture.attention.context.slot_mapping[layer.self_attn.attn.layer_name])
            for layer in primitive.model.layers
        )
        saved = tuple(cache[slot // cache.shape[2], :, slot % cache.shape[2]].clone() for cache, slot in slots)
        rng = torch.cuda.get_rng_state()
        native(graph)
        output = capture.output.clone()
        live_fields = tuple(value.clone() for value in ports)
        live = tuple(cache[slot // cache.shape[2], :, slot % cache.shape[2]].clone() for cache, slot in slots)
        for port, value in zip(ports, fields, strict=True):
            port.copy_(value)
        for (cache, slot), value in zip(slots, saved, strict=True):
            cache[slot // cache.shape[2], :, slot % cache.shape[2]] = value
        for adapter in capture.routes:
            adapter.reset()
        context = capture.attention.context

        def layer(ordinal, hidden, residual, kv="commit"):
            entries = hidden.clone(), None if residual is None else residual.clone()
            deltas = (
                tuple(
                    adapter.values(targets[ordinal - 22], adapter.private)["delta"].clone()
                    for adapter in capture.routes
                )
                if ordinal in (22, 23)
                else ()
            )
            inputs = []
            hook = primitive.model.layers[ordinal].self_attn.qkv_proj.register_forward_pre_hook(
                lambda module, args: inputs.append(args[0].clone())
            )
            try:
                actual = primitive.model.layers[ordinal](capture.positions, hidden, residual)
            finally:
                hook.remove()
            for lane, delta in enumerate(deltas):
                metadata = {
                    name: replace(
                        item, seq_lens=item.seq_lens[lane : lane + 1], block_table=item.block_table[lane : lane + 1]
                    )
                    for name, item in capture.attention.metadata.items()
                }
                lane_context = replace(context, attn_metadata=metadata)
                entry = entries[0][lane : lane + 1], None if entries[1] is None else entries[1][lane : lane + 1]
                expected, carry, kv = oracle_layer(
                    primitive.model.layers[ordinal],
                    entry,
                    lane_context,
                    capture.positions[lane : lane + 1],
                    delta,
                    inputs[0][lane : lane + 1],
                )
                relative = (
                    (actual[0][lane : lane + 1] + actual[1][lane : lane + 1]).float() - (expected + carry).float()
                ).norm() / (expected + carry).float().norm()
                assert relative < 0.02, float(relative)
                errors.append(float(relative))
                cache, slot = slots[ordinal]
                torch.testing.assert_close(
                    cache[slot[lane] // cache.shape[2], :, slot[lane] % cache.shape[2]], kv, rtol=0.02, atol=0.02
                )
            return actual

        # Independent explicit route, with the same physical GEMM width as the tested graph.
        with override_forward_context(context):
            hidden, residual, taps = capture.seed.clone(), None, []
            for ordinal in range(24):
                hidden, residual = layer(ordinal, hidden, residual)
                taps.append(hidden + residual)
            for adapter in capture.routes:
                adapter.apply_updates()
            for _ in range(workload.iterations):
                hidden, residual = taps[21].clone(), None
                for ordinal in (22, 23):
                    hidden, residual = layer(ordinal, hidden, residual)
            expected = primitive.model.norm(taps[-1])
        torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
        assert torch.equal(
            F.linear(output, handle.runner.model.lm_head.weight).argmax(-1),
            F.linear(expected, handle.runner.model.lm_head.weight).argmax(-1),
        )
        for port, value in zip(ports, live_fields, strict=True):
            torch.testing.assert_close(port, value, rtol=0.04, atol=0.04)
            port.copy_(value)
        for (cache, slot), value in zip(slots, live, strict=True):
            torch.testing.assert_close(
                cache[slot // cache.shape[2], :, slot % cache.shape[2]], value, rtol=0.04, atol=0.04
            )
            cache[slot // cache.shape[2], :, slot % cache.shape[2]] = value
        primitive.projections.prepare_wave(
            tuple(primitive.rows[adapter.row] for adapter in capture.routes), capture.routes, capture.lanes
        )
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng, rtol=0, atol=0)
        checked.append(capture.seed.shape[0])

    try:
        with patch.object(torch.cuda.CUDAGraph, "replay", replay):
            outputs = handle.generate(
                ["The sky is", "Water freezes at"], SamplingParams(max_tokens=3, temperature=0.0, ignore_eos=True)
            )
        assert len(outputs) == 2 and sum(checked) == 4
        assert len(primitive.captures) == 1

        def measured(function):
            def run(*args, **kwargs):
                with torch.profiler.record_function("route.forward"):
                    return function(*args, **kwargs)

            return run

        with (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as profile,
            patch.object(primitive.model, "forward", measured(primitive.forward)),
        ):
            profile_outputs = handle.generate(
                ["The sky is", "Water freezes at"], SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
            )
        assert any(event.name == "route.forward" for event in profile.events())
        for event in profile.events():
            parent = event.cpu_parent
            while parent is not None and parent.name != "route.forward":
                parent = parent.cpu_parent
            if parent is not None:
                assert not any(
                    word in event.name.lower()
                    for word in ("dtoh", "device to host", "synchronize", "local_scalar_dense")
                ), event.name
        diagnostics = handle.diagnostics()
        assert diagnostics.trained_rows == diagnostics.update_count * chunk and diagnostics.update_count == len(
            targets
        ) * sum(
            (len(output.prompt_token_ids) + len(output.outputs[0].token_ids) - 1) // chunk
            for output in (*outputs, *profile_outputs)
        )
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == 0
        print(
            f"route_updates chunk={chunk}: tokens={len(checked)} routed=4 variants=2 updates={diagnostics.update_count} trained_rows={diagnostics.trained_rows} relative_error={max(errors):.6f}"
        )
    finally:
        handle.close()


def main():
    for chunk in (2, 1):
        run(chunk)


if __name__ == "__main__":
    main()
