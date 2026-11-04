"""Explicit host lookup matches device lookup, neutral residuals and the bounded-memory large-table case."""

from dataclasses import replace

import torch
from safetensors.torch import save_file

import pllm
from examples.engram_lite import EngramLite, fit
from pllm.testing import check_on_gpu
from test.gpu.features.test_custom_residual import CONFIG, Observe, _generate


def test_engram_placements_and_capture(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("The capital of France is Paris. Water boils at a temperature of 100 degrees. " * 8)
    path = str(tmp_path / "memory.safetensors")
    fit(str(corpus), path, buckets=1024, width=16)
    host = EngramLite(path, buckets=1024, width=16, placement="host")
    config = replace(CONFIG, cuda_graphs=True, max_num_batched_tokens=32, enable_chunked_prefill=True)
    expected = _generate(replace(host, placement="device"), config)
    assert _generate(host, config) == expected
    assert _generate(host, replace(config, cuda_graphs=False)) == expected
    # Residual hooks capture vLLM source operators; use that arithmetic for the logprob oracle.
    assert _generate(replace(host, gate=0), config) == _generate(Observe(), replace(config, cuda_graphs=False))
    result = check_on_gpu(host, replace(config, max_num_batched_tokens=128), neutral=replace(host, gate=0))
    assert "host syncs: eight steady decode steps" in result.passed
    assert (
        next(
            task for member in pllm.explain(host, config).members for task in member.tasks if task.name == "lookup"
        ).lane
        == "side"
    )


def test_three_gib_host_table_with_one_gib_device_budget(tmp_path):
    path = str(tmp_path / "large.safetensors")
    buckets, width = 3 * 1024**3 // 64, 32
    table = torch.zeros(buckets, width, dtype=torch.bfloat16)
    weights = torch.ones(2 * width, 896, dtype=torch.bfloat16) / 64
    save_file({"table": table, "wk": weights, "wv": weights.clone()}, path)
    del table
    workload = EngramLite(path, buckets=buckets, placement="host")
    config = replace(CONFIG, cuda_graphs=True, execution=replace(CONFIG.execution, memory_budget_bytes=1 << 30))
    report = pllm.explain(workload, config)
    assert report.peak_memory_bytes < 1 << 30
    assert next(state for state in report.states if state.name == "memory").host_bytes >= 3 * 1024**3
    assert _generate(workload, config) == _generate(Observe(), replace(config, cuda_graphs=False))
