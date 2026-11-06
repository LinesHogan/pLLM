"""Qwen GRPO: two on-policy host rounds and the GPU checker's aligned sequence permutation."""

from __future__ import annotations

# Diagnostics inspect the in-process engine, as in pllm.testing_gpu.
# pyright: reportPrivateUsage=false
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from vllm import RequestOutput, SamplingParams

import pllm
from pllm.contracts.packed_rows import PackedRows
from pllm.testing import check_on_gpu
from test.workloads.grpo_workload import GRPO


class TokenRewardGRPO(GRPO):
    """A deterministic token-score verifier keeps the smoke independent of answer formatting."""

    def rewards(self, output: RequestOutput) -> torch.Tensor:
        """Prefer samples with smaller token IDs, producing nonconstant group rewards."""
        return torch.tensor([-float(sum(sample.token_ids)) for sample in output.outputs])


def main() -> None:
    """Run the checker and reuse TTRL's sampling/feedback loop for exactly two GRPO updates."""
    config = pllm.ModelConfig(
        "Qwen/Qwen2.5-0.5B-Instruct",
        max_model_len=64,
        max_num_seqs=4,
        max_num_batched_tokens=64,
        gpu_memory_utilization=0.35,
        seed=7,
        execution=pllm.ExecutionOptions(max_training_rows=8, publish_lag=0),
    )
    workload = TokenRewardGRPO()
    report = pllm.explain(workload, config)
    assert "CausalTrainBatch (input_ids, labels, attention_mask, weights)" in str(report)
    result = check_on_gpu(workload, config)
    assert any("sequence permutation" in name for name in result.passed)
    print(result)
    with pllm.make_llm(config, workload=workload) as handle:
        execution = handle.runner.require_session()._deployment.execution
        (initial,) = execution.directory.select(PackedRows.from_rows(())).refs
        before = execution.snapshot(initial, torch.cuda.current_stream())[0].cpu()
        batches = workload.run(
            handle,
            ["Choose a random digit: ", "A random number is "],
            SamplingParams(n=4, temperature=1.0, max_tokens=4, ignore_eos=True, seed=17),
            rounds=2,
            max_rows=8,
            pad_token=0,
        )
        assert all(batch is not None for batch in batches)
        for batch in batches:
            assert batch is not None and batch.weights is not None
            assert batch.weights.min() < 0 < batch.weights.max()
        diagnostics = handle.diagnostics()
        assert diagnostics.update_count == diagnostics.loss_count == 2
        (current,) = execution.directory.select(PackedRows.from_rows(())).refs
        assert current.version_id == initial.version_id + 2
        after = execution.snapshot(current, torch.cuda.current_stream())[0].cpu()
        assert not torch.equal(after, before)
        print(f"GRPO Qwen2.5-0.5B: rounds=2 updates={diagnostics.update_count}, TTRL host loop reused")
    assert execution.diagnostics().active_states == 0


if __name__ == "__main__":
    main()
