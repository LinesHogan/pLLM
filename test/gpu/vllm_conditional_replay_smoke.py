"""Evidence 41: real layer-range recurrence, per-layer current-slot KV and preserved normal output."""

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F
from vllm import SamplingParams

import pllm
from pllm.backends.vllm.replay import VllmLayerReplay
from pllm.backends.vllm.runtime import WorkloadModelRunner, probe_backend
from pllm.config import ModelConfig, VllmDType
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
from pllm.contracts.semantic_graph import ConditionalLayerReplay
from pllm.execution.replay import ReplayStateTask
from pllm.execution.trace import TracePhase
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload


def main():
    targets = tuple(
        ParameterTarget(layer, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT) for layer in (22, 23)
    )
    workload = InPlaceTTTWorkload(
        InPlaceTTTConfig(targets, chunk_size=2, learning_rate=1e-6, replay=ConditionalLayerReplay((22, 23)))
    )
    original_attach = WorkloadModelRunner.attach_workload

    def checked_attach(runner, deployment):
        timeline = deployment.attachment.inputs.plan.timeline
        assert min(timeline.reserved_bytes) > 0
        facts = probe_backend(workload.backend_requirements(), runner, timeline)
        for index in range(len(timeline.anchors)):
            short = replace(
                timeline,
                reserved_bytes=tuple(value - int(i == index) for i, value in enumerate(timeline.reserved_bytes)),
            )
            underfunded = workload.compile_backend(replace(facts, timeline=short))
            try:
                original_attach(runner, underfunded)
            except ValueError as error:
                assert "effect buffers" in str(error)
            else:
                raise AssertionError("worker accepted a re-signed underfunded effect plan")
        original_attach(runner, deployment)

    with patch.object(WorkloadModelRunner, "attach_workload", checked_attach):
        handle = pllm.make_llm(
            workload=workload,
            config=ModelConfig("Qwen/Qwen2.5-0.5B-Instruct", VllmDType.BFLOAT16, 64, 2, 64, 0.25, seed=7),
        )
    layers = tuple(handle.runner.model.model.layers[index] for index in (22, 23))
    normal_outputs, events, comparisons = [], [], []
    original_task, original_step = ReplayStateTask.execute, VllmLayerReplay.step

    def remember(module, args, result):
        normal_outputs.append(result)

    def rms(value, norm):
        return (
            value.float()
            * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + norm.variance_epsilon)
            * norm.weight.float()
        ).to(norm.weight.dtype)

    bindings = handle.deployment.attachment.inputs.request.overlay_bindings

    def oracle_layer(layer, layer_index, parts, item, arena, pin, ref):
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
        metadata = item.metadata[attention.attn.layer_name]
        block_table = metadata.block_table[0]
        length, block_size = metadata.max_seq_len, cache.shape[2]
        # Copy only valid history, not the model's large mostly unused KV allocation.
        history = torch.stack(
            [cache[int(block_table[pos // block_size]), :, pos % block_size] for pos in range(length)]
        )
        qkv = F.linear(hidden, attention.qkv_proj.weight, attention.qkv_proj.bias)
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], -1)
        q, k, v = (value.reshape(-1, attention.head_dim) for value in (q, k, v))
        cos, sin = attention.rotary_emb.cos_sin_cache[int(item.positions.item())].chunk(2)

        def rotate(value):
            first, second = value.chunk(2, -1)
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
        (binding,) = (binding for binding in bindings if binding.target.layer == layer_index)
        delta = arena.borrow_field(pin, ref, binding.fields[0])
        expected = F.linear(hidden, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias)
        expected += F.linear(hidden, delta)
        return expected, residual, cache, expected_kv, history[:-1].clone()

    def check_step(primitive, row, carry, arena, pin, ref):
        item = next(item for item in primitive._inputs if item.row is row)
        # The fused add-RMSNorm mutates its inputs in place; clone chain hops at
        # call entry, before each layer overwrites its hidden/residual buffers.
        captured = []

        def spy(original):
            def wrapped(positions, hidden, residual=None):
                entry = (hidden.clone(), None if residual is None else residual.clone())
                out = original(positions, hidden, residual)
                captured.append((entry, out[0].clone(), out[1].clone()))
                return out

            return wrapped

        saved = primitive.originals
        primitive.originals = tuple(spy(original) for original in saved)
        try:
            actual = original_step(primitive, row, carry, arena, pin, ref)
        finally:
            primitive.originals = saved
        # Feed each layer's oracle its real replayed input pair, keeping every
        # layer's numerical bound as tight as a single-layer replay.
        for (layer_index, layer), ((entry_hidden, entry_residual), real_hidden, real_residual), (
            slot_cache,
            block,
            offset,
        ) in zip(((22, layers[0]), (23, layers[1])), captured, item.slots, strict=True):
            mlp_out, residual, cache, expected_kv, historical_kv = oracle_layer(
                layer, layer_index, (entry_hidden, entry_residual), item, arena, pin, ref
            )
            expected = mlp_out + residual
            # Residual cancellation can amplify a one-ULP difference relative to
            # the layer output; bound rounding against the added operands. The
            # oracle's attention differs from FA2 by ~1 ULP of its own output;
            # where the residual stream is large that flips its bf16 rounding,
            # and the RMSNorm scale plus GEMM mixing spread that one-ULP flip to
            # every output element — hence the global stream-magnitude term.
            stream = entry_hidden.float().abs().max()
            if entry_residual is not None:
                stream = stream + entry_residual.float().abs().max()
            rounding_bound = (
                2 * torch.finfo(carry.dtype).eps * (mlp_out.float().abs() + residual.float().abs() + stream) + 0.02
            )
            error = ((real_hidden + real_residual).float() - expected.float()).abs()
            assert torch.all(error <= rounding_bound), (layer_index, error.max().item(), rounding_bound.max().item())
            relative = error.norm() / expected.float().norm()
            assert relative < 0.01, relative.item()
            assert cache is slot_cache
            torch.testing.assert_close(cache[block, :, offset], expected_kv, rtol=0.02, atol=0.02)
            metadata = item.metadata[layer.self_attn.attn.layer_name]
            block_table, length, block_size = metadata.block_table[0], metadata.max_seq_len, cache.shape[2]
            current_history = torch.stack(
                [cache[int(block_table[pos // block_size]), :, pos % block_size] for pos in range(length - 1)]
            )
            torch.testing.assert_close(current_history, historical_kv, rtol=0, atol=0)
        comparisons.append((row.identity, ref))
        return actual

    def check_task(task, context):
        primitive = context.replay
        slots = tuple(
            (item, tuple(cache[block, :, offset].clone() for cache, block, offset in item.slots))
            for item in primitive._inputs
        )
        for item, normal_kvs in slots:
            for original, normal_kv in zip(item.original_kv, normal_kvs, strict=True):
                torch.testing.assert_close(original, normal_kv, rtol=0, atol=0)
        rng = torch.cuda.get_rng_state()
        saved_outputs = tuple(tuple(value.clone() for value in output) for output in normal_outputs)
        original_task(task, context)
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng, rtol=0, atol=0)
        for output, saved in zip(normal_outputs, saved_outputs):
            for actual, expected in zip(output, saved):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        normal_outputs.clear()
        replay = tuple(
            event
            for event in context.trace.snapshot().events
            if event.invocation_id == context.invocation_id and event.phase is TracePhase.REPLAY
        )
        if context.updated_refs:
            assert context.published_pin is not None
            assert len(replay) <= 2 * len(context.updated_refs)
            assert any(
                not torch.equal(cache[block, :, offset], before)
                for item, befores in slots
                for (cache, block, offset), before in zip(item.slots, befores, strict=True)
            )
        else:
            assert context.published_pin is None and not replay
        events.extend(replay)

    hook = layers[1].register_forward_hook(remember)
    try:
        with patch.object(ReplayStateTask, "execute", check_task), patch.object(VllmLayerReplay, "step", check_step):
            outputs = handle.generate(
                ["The sky is", "Water freezes at"], SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True)
            )
        assert len(outputs) == 2 and len({identity for identity, ref in comparisons}) == 2
        assert len(comparisons) == len(events) == 4
        diagnostics = handle.deployment.execution.diagnostics()
        assert diagnostics.update_count == 4 and diagnostics.trained_rows == 8
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        print(
            dict(
                updates=diagnostics.update_count,
                replay_iterations=len(events),
                numerical_comparisons=len(comparisons),
                fallback=0,
            )
        )
    finally:
        hook.remove()
        handle.close()


if __name__ == "__main__":
    main()
