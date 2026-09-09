"""Evidence 41: real final-layer recurrence, current-slot KV and preserved normal output."""

import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from pllm.execution.trace import TracePhase
from torch.nn import functional as F
from vllm import SamplingParams

import pllm
from pllm.execution.replay import ReplayStateTask
from pllm.workloads.inplace_ttt import InPlaceTTTConfig, InPlaceTTTWorkload
from pllm.backends.vllm.replay import VllmLayerReplay
from pllm.backends.vllm.runtime import WorkloadModelRunner, probe_backend
from pllm.config import ModelConfig, VllmDType
from pllm.contracts.parameter_overlay import ParameterModuleKind, ParameterRole, ParameterTarget
from pllm.contracts.semantic_graph import ConditionalLayerReplay


def main():
    target = ParameterTarget(23, ParameterModuleKind.MLP, ParameterRole.DOWN_PROJECTION_WEIGHT)
    workload = InPlaceTTTWorkload(InPlaceTTTConfig((target,), chunk_size=2, learning_rate=1e-6,
        replay=ConditionalLayerReplay(23)))
    original_attach = WorkloadModelRunner.attach_workload
    def checked_attach(runner, deployment):
        timeline = deployment.attachment.inputs.plan.timeline
        assert min(timeline.reserved_bytes) > 0
        facts = probe_backend(workload.backend_requirements(), runner, timeline)
        for index in range(len(timeline.anchors)):
            short = replace(timeline, reserved_bytes=tuple(value - int(i == index)
                for i, value in enumerate(timeline.reserved_bytes)))
            underfunded = workload.compile_backend(replace(facts, timeline=short))
            try:
                original_attach(runner, underfunded)
            except ValueError as error:
                assert "effect buffers" in str(error)
            else:
                raise AssertionError("worker accepted a re-signed underfunded effect plan")
        original_attach(runner, deployment)
    with patch.object(WorkloadModelRunner, "attach_workload", checked_attach):
        handle = pllm.make_llm(workload=workload, config=ModelConfig(
            "Qwen/Qwen2.5-0.5B-Instruct", VllmDType.BFLOAT16, 64, 2, 64, 0.25, seed=7))
    layer = handle.runner.model.model.layers[23]
    normal_outputs, events, comparisons = [], [], []
    original_task, original_step = ReplayStateTask.execute, VllmLayerReplay.step

    def remember(module, args, result):
        normal_outputs.append(result)

    def rms(value, norm):
        return (value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True)
            + norm.variance_epsilon) * norm.weight.float()).to(norm.weight.dtype)

    def check_step(primitive, row, carry, arena, pin, ref):
        item = next(item for item in primitive._inputs if item.row is row)
        attention = layer.self_attn
        cache = attention.attn.kv_cache
        block_table = item.metadata.block_table[0]
        length, block_size = item.metadata.max_seq_len, cache.shape[2]
        # Copy only valid history, not the model's large mostly unused KV allocation.
        history = torch.stack([cache[int(block_table[pos // block_size]), :, pos % block_size]
            for pos in range(length)])
        hidden = rms(carry, layer.input_layernorm)
        qkv = F.linear(hidden, attention.qkv_proj.weight, attention.qkv_proj.bias)
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], -1)
        q, k, v = (value.reshape(-1, attention.head_dim) for value in (q, k, v))
        cos, sin = attention.rotary_emb.cos_sin_cache[int(item.positions.item())].chunk(2)
        def rotate(value):
            first, second = value.chunk(2, -1)
            return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(carry.dtype)
        q, k = rotate(q), rotate(k)
        expected_kv = torch.cat((k, v), -1)
        history[-1] = expected_kv
        keys, values = history.chunk(2, -1)
        repeat = attention.num_heads // attention.num_kv_heads
        keys, values = (value.repeat_interleave(repeat, dim=1).transpose(0, 1).float() for value in (keys, values))
        scores = (q.float().unsqueeze(1) * keys).sum(-1) * attention.scaling
        attended = (scores.softmax(-1).unsqueeze(-1) * values).sum(1).reshape(1, -1).to(carry.dtype)
        hidden = F.linear(attended, attention.o_proj.weight, attention.o_proj.bias)
        residual_fp32 = carry.float() + hidden.float()
        residual = residual_fp32.to(carry.dtype)
        hidden = rms(residual_fp32, layer.post_attention_layernorm)
        gate, up = F.linear(hidden, layer.mlp.gate_up_proj.weight, layer.mlp.gate_up_proj.bias).chunk(2, -1)
        hidden = (F.silu(gate.float()) * up.float()).to(carry.dtype)
        binding, = handle.deployment.attachment.inputs.request.overlay_bindings
        delta = arena.borrow_field(pin, ref, binding.fields[0])
        expected = F.linear(hidden, layer.mlp.down_proj.weight, layer.mlp.down_proj.bias)
        expected += F.linear(hidden, delta)
        rounding_bound = 2 * torch.finfo(carry.dtype).eps * (expected.float().abs() + residual.float().abs()) + 0.02
        expected = expected + residual
        historical_kv = history[:-1].clone()
        actual = original_step(primitive, row, carry, arena, pin, ref)
        # Residual cancellation can amplify a one-ULP difference relative to the
        # final output; bound rounding against the added operands, not that output.
        error = (actual.float() - expected.float()).abs()
        assert torch.all(error <= rounding_bound), (error.max().item(), rounding_bound.max().item())
        relative = error.norm() / expected.float().norm()
        assert relative < 0.01, relative.item()
        torch.testing.assert_close(cache[item.block, :, item.offset], expected_kv, rtol=0.02, atol=0.02)
        current_history = torch.stack([cache[int(block_table[pos // block_size]), :, pos % block_size]
            for pos in range(length - 1)])
        torch.testing.assert_close(current_history, historical_kv, rtol=0, atol=0)
        comparisons.append((row.identity, ref))
        return actual

    def check_task(task, context):
        primitive = context.replay
        slots = tuple((item, layer.self_attn.attn.kv_cache[item.block, :, item.offset].clone())
            for item in primitive._inputs)
        for item, normal_kv in slots:
            torch.testing.assert_close(item.original_kv, normal_kv, rtol=0, atol=0)
        rng = torch.cuda.get_rng_state()
        saved_outputs = tuple(tuple(value.clone() for value in output) for output in normal_outputs)
        original_task(task, context)
        torch.testing.assert_close(torch.cuda.get_rng_state(), rng, rtol=0, atol=0)
        for output, saved in zip(normal_outputs, saved_outputs):
            for actual, expected in zip(output, saved):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        normal_outputs.clear()
        replay = tuple(event for event in context.trace.snapshot().events
            if event.invocation_id == context.invocation_id and event.phase is TracePhase.REPLAY)
        if context.updated_refs:
            assert context.published_pin is not None
            assert len(replay) <= 2 * len(context.updated_refs)
            assert any(not torch.equal(layer.self_attn.attn.kv_cache[item.block, :, item.offset], before)
                for item, before in slots)
        else:
            assert context.published_pin is None and not replay
        events.extend(replay)

    hook = layer.register_forward_hook(remember)
    try:
        with patch.object(ReplayStateTask, "execute", check_task), patch.object(VllmLayerReplay, "step", check_step):
            outputs = handle.generate(["The sky is", "Water freezes at"],
                SamplingParams(max_tokens=2, temperature=0.0, ignore_eos=True))
        assert len(outputs) == 2 and len({identity for identity, ref in comparisons}) == 2
        assert len(comparisons) == len(events) == 4
        diagnostics = handle.deployment.execution.diagnostics()
        assert diagnostics.update_count == 2 and diagnostics.trained_rows == 4
        assert diagnostics.active_lineages == diagnostics.pending_state_releases == diagnostics.fallback_count == 0
        print(dict(updates=diagnostics.update_count, replay_iterations=len(events), numerical_comparisons=len(comparisons), fallback=0))
    finally:
        hook.remove()
        handle.close()


if __name__ == "__main__":
    main()
