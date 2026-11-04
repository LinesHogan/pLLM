"""Native LSE bitwise parity, captured monitors and scored dense read selection."""

from dataclasses import replace

import pytest
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

import pllm
from pllm.execution.attention import DenseAttention, PagedKV
from pllm.execution.attention_methods import TraceRead
from pllm.testing import check_on_gpu
from test.workloads.attention_workloads import InstanceAttention, Monitor, Quest, page_bounds

CONFIG = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=128,
    max_num_seqs=4,
    max_num_batched_tokens=128,
    gpu_memory_utilization=0.3,
    cuda_graphs=True,
    execution=pllm.ExecutionOptions(capture_memory_bytes=256 << 20),
)


@pytest.mark.parametrize("workload", (Monitor(), Quest(), InstanceAttention()))
def test_attention_contracts(workload):
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
