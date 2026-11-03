"""Explanation costs compared against loaded Qwen2.5 probes for sampling and prompt-produced LoRA."""

import pytest

import pllm
from pllm.workloads import ESampConfig, ESampWorkload, PromptFactorsWorkload
from test.explain_probe import assert_explanation


@pytest.mark.parametrize(
    "workload",
    (
        ESampWorkload(ESampConfig(rank=2, intervention_beta=0.5)),
        PromptFactorsWorkload(rank=2),
    ),
)
def test_explain_matches_loaded_probe(workload):
    config = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=64,
        max_num_seqs=2,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.25,
    )
    report = pllm.explain(workload, config)
    with pllm.make_llm(config, workload=workload) as handle:
        assert_explanation(workload, config, handle)
        assert report == pllm.explain(workload, config)
