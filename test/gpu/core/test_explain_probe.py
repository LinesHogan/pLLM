"""Explanation costs compared against loaded Qwen2.5 probes for sampling and prompt-produced LoRA."""

import pytest
import torch

import pllm
from pllm.testing_gpu import check_probe
from pllm.workloads import ESampConfig, ESampWorkload

pytestmark = pytest.mark.gpu


def test_explain_matches_loaded_probe(monkeypatch):
    workload = ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5))
    config = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
    )
    report = pllm.explain(workload, config)
    with pllm.make_llm(config, workload=workload) as handle:
        with monkeypatch.context() as capacity:
            capacity.setattr(torch.cuda, "mem_get_info", lambda device=None: (1, 1))
            check_probe(workload, config, handle)
        assert report == pllm.explain(workload, config)
        diagnostics = handle.diagnostics()
        assert diagnostics.kv_cache_blocks > 0 and diagnostics.kv_cache_max_concurrency > 0
