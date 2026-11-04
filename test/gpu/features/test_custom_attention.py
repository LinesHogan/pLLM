"""Native LSE bitwise parity, captured monitors and scored dense read selection."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

import pllm
from pllm.execution.attention import DenseAttention, PagedKV
from pllm.execution.attention_methods import TraceRead
from pllm.testing import check_on_gpu
from test.workloads.attention_workloads import (
    InstanceAttention,
    Monitor,
    Quest,
    SharedAttention,
    SharedWindowAttention,
    SinkAttention,
    WindowAttention,
    page_bounds,
)

CONFIG = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=128,
    max_num_seqs=4,
    max_num_batched_tokens=128,
    gpu_memory_utilization=0.3,
    cuda_graphs=True,
    execution=pllm.ExecutionOptions(capture_memory_bytes=256 << 20),
)


@pytest.mark.parametrize(
    "workload",
    (
        Monitor(),
        Quest(),
        InstanceAttention(),
        WindowAttention(),
        SinkAttention(),
        SharedAttention(),
        SharedWindowAttention(),
    ),
)
def test_attention_contracts(workload, monkeypatch):
    if any(layout.keep == "none" for layout in workload.backend_requirements().kv_layouts):
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    result = check_on_gpu(workload, CONFIG, neutral=workload if isinstance(workload, Monitor) else None)
    assert any("host syncs" in name for name in result.passed)
    if isinstance(workload, Monitor):
        from vllm import SamplingParams

        with pllm.make_llm(CONFIG, workload=workload) as engine:
            engine.select_trace(pllm.io.TraceSelection(tensors=True))
            engine.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0, ignore_eos=True))
            events = engine.trace_snapshot().events
            mass = [event.tensor() for event in events if event.tensor_name == "recent_mass"]
            assert mass and all(torch.all((value > 0) & (value <= 1.001)) for value in mass)


def test_native_lse_and_selected_oracle():
    torch.manual_seed(41)
    heads, kv_heads, dim, block = 14, 2, 64, 16
    cache = torch.randn(16, kv_heads, block, 2 * dim, device="cuda", dtype=torch.bfloat16)
    keys, values = cache.transpose(1, 2).chunk(2, -1)
    table = torch.randperm(16, device="cuda").int().reshape(2, 8)
    lengths = torch.tensor([95, 81], device="cuda", dtype=torch.int32)
    for queries in ((7, 9), (1, 1), (1, 3)):
        q = torch.randn(sum(queries), heads, dim, device="cuda", dtype=torch.bfloat16)
        starts = torch.tensor([0, queries[0], sum(queries)], device="cuda", dtype=torch.int32)
        args = dict(
            q=q,
            k=keys,
            v=values,
            cu_seqlens_q=starts,
            max_seqlen_q=max(queries),
            seqused_k=lengths,
            max_seqlen_k=95,
            softmax_scale=dim**-0.5,
            causal=True,
            block_table=table,
            fa_version=2,
        )
        native = torch.empty_like(q)
        flash_attn_varlen_func(**args, out=native)
        eager, lse = flash_attn_varlen_func(**args, return_softmax_lse=True)
        assert torch.equal(native, eager)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured, captured_lse = flash_attn_varlen_func(**args, return_softmax_lse=True)
        graph.replay()
        assert torch.equal(native, captured) and torch.equal(lse, captured_lse)
        positions = torch.cat(
            tuple(torch.arange(length - query, length, device="cuda") for length, query in zip((95, 81), queries))
        )
        kv = PagedKV(keys, values, table, lengths, starts, positions, TraceRead(128))
        kv = replace(kv, _read=DenseAttention(kv, dim**-0.5))
        scores = page_bounds(q, keys, table, starts, positions)
        torch.testing.assert_close(
            scores, page_bounds(q.cpu(), keys.cpu(), table.cpu(), starts.cpu(), positions.cpu()), check_device=False
        )
        selected, _ = Quest().select(q, kv, layer=0)
        pages = scores.topk(2, -1).indices.cpu()
        expected = torch.empty_like(q, dtype=torch.float32)
        for row in range(len(q)):
            request = int(row >= queries[0])
            for head in range(heads):
                chosen = (pages[row, head, : 1 + head % 2, None] * block + torch.arange(block)).flatten()
                chosen = chosen[chosen <= int(positions[row])].to("cuda")
                physical = table[request, chosen // block].long()
                k = keys[physical, chosen % block, head // (heads // kv_heads)].float()
                v = values[physical, chosen % block, head // (heads // kv_heads)].float()
                probability = (q[row, head].float() @ k.T * dim**-0.5).softmax(-1)
                expected[row, head] = probability @ v
        torch.testing.assert_close(selected.float(), expected, atol=0.008, rtol=0.008)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_selected, _ = Quest().select(q, kv, layer=0)
        graph.replay()
        assert torch.equal(selected, captured_selected)


def _layout_read(read, errors, kernel, q, window, topk):
    """Compare native reads with a dense FP32 oracle while pages are being recycled."""
    mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        out, lse = read(kernel, q, window, topk)
    finally:
        torch.cuda.set_sync_debug_mode(mode)
    if kernel.metadata.max_query_len > 1 or kernel.metadata.max_seq_len > 96:
        kv = kernel.kv
        # Independent FP32 softmax over logical positions. No production masking/gather helper.
        for row in range(len(q)):
            request = sum(row >= int(start) for start in kv.query_start[1:])
            position = int(kv.positions[row])
            chosen = torch.arange(position + 1, device=q.device)
            keep = torch.ones_like(chosen, dtype=torch.bool)
            if kv.storage_window:
                keep = chosen > position - kv.storage_window
                if kv.sink_lens is not None:
                    keep |= chosen < kv.sink_lens[row]
            if window:
                keep &= chosen > position - window
            chosen = chosen[keep]
            blocks = kv.block_table[request, chosen // kv.key_cache.shape[1]].long()
            k, v = (
                cache[blocks, chosen % cache.shape[1]]
                .float()
                .repeat_interleave(q.shape[1] // cache.shape[2], 1)
                .transpose(0, 1)
                for cache in (kv.key_cache, kv.value_cache)
            )
            scores = (q[row].float()[:, None] @ k.transpose(1, 2)).squeeze(1) * q.shape[-1] ** -0.5
            expected = (scores.softmax(-1)[:, None] @ v).squeeze(1)
            torch.testing.assert_close(out[row].float(), expected, atol=0.016, rtol=0.016)
            errors.append(float((out[row].float() - expected).abs().max()))
    return out, lse


@pytest.mark.parametrize(
    "workload", (WindowAttention(), SinkAttention(), SharedAttention(tuple(range(1, 24, 2))), SharedWindowAttention())
)
def test_layout_oracle_and_reclaimed_blocks(workload):
    from vllm import SamplingParams

    from pllm.backends.vllm.custom_attention import FlashRead
    from pllm.contracts.kv_layout import storage_layout

    config = replace(CONFIG, cuda_graphs=False)
    errors = []
    read = FlashRead.__call__

    def oracle(kernel, q, window, topk):
        return _layout_read(read, errors, kernel, q, window, topk)

    with pllm.make_llm(config, workload=workload) as engine:
        runner = engine.runner
        model = runner.decoder.kv_writable("layout oracle")
        program = runner.require_session()._deployment.execution.math.attention
        # The native cache tensors alias exactly; the consumer's K/V write is skipped by vLLM.
        for layout in workload.backend_requirements().kv_layouts:
            if layout.keep == "none":
                target = storage_layout(program.layouts, layout.layer).layer
                assert (
                    model.model.layers[layout.layer].self_attn.attn.kv_cache.data_ptr()
                    == model.model.layers[target].self_attn.attn.kv_cache.data_ptr()
                )
        manager = engine._require_open().llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
        allocated = manager.allocate_slots
        measurements = []

        def allocate(request, *args, **kwargs):
            result = allocated(request, *args, **kwargs)
            if request.num_computed_tokens >= 112:
                for group, native in zip(
                    manager.kv_cache_config.kv_cache_groups, manager.coordinator.single_type_managers
                ):
                    blocks = native.req_to_blocks[request.request_id]
                    held = sum(block.block_id != 0 for block in blocks)
                    measurements.append((group.layer_names, held, len(blocks)))
            return result

        with patch.object(FlashRead, "__call__", oracle), patch.object(manager, "allocate_slots", allocate):
            engine.generate(
                [{"prompt_token_ids": [15, 16, 17, 18, 19, 20, 21, 22]}],
                SamplingParams(max_tokens=112, temperature=0, ignore_eos=True),
            )
        assert errors and measurements
        if isinstance(workload, (WindowAttention, SinkAttention)):
            layers = {model.model.layers[index].self_attn.attn.layer_name for index in (0, 1)}
            retained = [(held, full) for names, held, full in measurements if layers.intersection(names)]
            assert retained and all(held < full for held, full in retained)
            print(type(workload).__name__, "allocated/full blocks:", retained[-1], "oracle max error:", max(errors))
        else:
            owned = sum(len(group.layer_names) for group in manager.kv_cache_config.kv_cache_groups)
            assert owned * 2 == len(model.model.layers)
            print(
                "SharedAttention allocated layers:",
                owned,
                "of",
                len(model.model.layers),
                "oracle max error:",
                max(errors),
            )
