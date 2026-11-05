"""Explicit host lookup matches device lookup, neutral residuals and the bounded-memory large-table case."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

import pllm
from examples.engram_lite import EngramLite, fit
from pllm.execution.state_execution import StateExecution
from pllm.testing import check_on_gpu
from pllm.testing_gpu import _release_models
from test.gpu.features.test_custom_residual import CONFIG, Observe, _generate
from test.workloads.test_pretrained import LogitLookup


def generation(workload, config):
    """Drain closed compiled models before vLLM takes its startup memory snapshot."""
    _release_models()
    return _generate(workload, config)


@pytest.mark.parametrize("logits", (False, True))
def test_engram_placements_and_capture(tmp_path, logits):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("The capital of France is Paris. Water boils at a temperature of 100 degrees. " * 8)
    path = str(tmp_path / "memory.safetensors")
    fit(str(corpus), path, buckets=1024, width=16)
    host = EngramLite(path, buckets=1024, width=16, placement="host")
    if logits:
        save_file({"table": torch.randn(8, 151936, dtype=torch.bfloat16) / 64}, path)
        host = LogitLookup(path)
    config = replace(CONFIG, cuda_graphs=True, max_num_batched_tokens=32, enable_chunked_prefill=True)
    expected = generation(replace(host, placement="device"), config)
    execute = StateExecution._execute_task

    def delayed(execution, context, task):
        if task.lane == "side":
            with torch.cuda.stream(execution.side_stream):
                torch.cuda._sleep(10_000_000)
        execute(execution, context, task)

    # The main model reaches the consumer before the side copy unless its graph waits on the event.
    with patch.object(StateExecution, "_execute_task", delayed):
        assert generation(host, config) == expected
    if not logits:
        assert generation(host, replace(config, cuda_graphs=False)) == expected
    # Residual hooks capture vLLM source operators; use that arithmetic for the logprob oracle.
    baseline = generation(None, config) if logits else generation(Observe(), replace(config, cuda_graphs=False))
    assert generation(replace(host, gate=0), config) == baseline
    result = check_on_gpu(host, replace(config, max_num_batched_tokens=128), neutral=replace(host, gate=0))
    assert "host syncs: eight steady decode steps" in result.passed
    assert (
        next(
            task for member in pllm.explain(host, config).members for task in member.tasks if task.name == "lookup"
        ).lane
        == "side"
    )
    if logits:
        report = pllm.explain(host, config)
        assert next(state for state in report.states if state.name == "memory").transfer_bytes_per_row == 2 * 151936


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
    assert generation(workload, config) == generation(Observe(), replace(config, cuda_graphs=False))
