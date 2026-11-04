"""User custom operators in residual hooks and forward cells: graphs, syncs and CPU/CUDA agreement."""

import os
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

import pllm
from pllm.testing import check_on_gpu
from test.workloads.kernel_workloads import Forward, Residual, add, scale

CONFIG = pllm.ModelConfig(
    "Qwen/Qwen2.5-0.5B-Instruct",
    max_model_len=128,
    max_num_seqs=4,
    max_num_batched_tokens=128,
    # These tiny schedules need little KV; leave room for the resident FP32 oracle on a 24 GiB GPU.
    gpu_memory_utilization=0.2,
    cuda_graphs=True,
    execution=pllm.ExecutionOptions(capture_memory_bytes=256 << 20),
)


WORKLOADS = (Residual(), Forward(), Forward(constant=True))


@pytest.mark.parametrize("index", range(len(WORKLOADS)), ids=("residual", "forward", "constant"))
def test_custom_kernels(index):
    # Keep native batch-invariant GEMMs out of the parent process's FP32 oracle tests.
    subprocess.run(
        [sys.executable, __file__, str(index)],
        env={**os.environ, "VLLM_BATCH_INVARIANT": "1"},
        check=True,
    )


def _kernel_contract(workload):
    config = replace(CONFIG, cuda_graphs="auto") if isinstance(workload, Forward) else CONFIG
    neutral = None if isinstance(workload, Forward) and workload.constant else workload
    result = check_on_gpu(workload, config, neutral=neutral)
    assert any("device equivalence" in name for name in result.passed)
    assert any("host syncs" in name for name in result.passed)
    x = torch.randn(3, 896, device="cuda", dtype=torch.bfloat16)
    y = torch.randn_like(x)
    eager = scale(add(x, y))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = scale(add(x, y))
    graph.replay()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def test_wrong_reference_fails_device_equivalence():
    @add.register_kernel("cpu")
    def wrong(x, y):
        return x + y + 1

    try:
        # Fail before an engine is loaded; the same helper is check_on_gpu's first contract.
        from pllm.testing_contracts import _check

        with pytest.raises(AssertionError, match="device equivalence"):
            _check(Residual(), replace(CONFIG, cuda_graphs=False), device_equivalence=True)
    finally:
        from test.workloads.kernel_workloads import _cpu

        add.register_kernel("cpu")(_cpu)


if __name__ == "__main__":
    _kernel_contract(WORKLOADS[int(sys.argv[1])])
