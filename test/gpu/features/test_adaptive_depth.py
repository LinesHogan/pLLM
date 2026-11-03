"""Native conditional-depth contracts in isolated engines, in both supported vLLM environments."""

import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="conditional decoder graphs need CUDA")
@pytest.mark.parametrize(
    "rule,batch,value",
    (("all", 3, False), ("none", 8, False), ("mixed", 8, False), ("none", 3, True), ("buckets", 8, False)),
)
def test_conditional_depth_native_kv_and_sync_debug(rule, batch, value):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "test.gpu.vllm_adaptive_depth_smoke",
            "--rule",
            rule,
            "--batch",
            str(batch),
            "--tokens",
            "12" if rule == "buckets" else "4",
            "--gpu-memory-utilization",
            "0.5",
            *(["--value-rewrite"] if value else []),
        ],
        check=True,
    )
