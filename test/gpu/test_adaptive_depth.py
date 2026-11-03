"""Native conditional-depth contracts in isolated engines, in both supported vLLM environments."""

import subprocess
import sys

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="conditional decoder graphs need CUDA")
@pytest.mark.parametrize("rule,batch", (("all", 3), ("none", 8), ("mixed", 8)))
def test_conditional_depth_native_kv_and_sync_debug(rule, batch):
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
            "4",
        ],
        check=True,
    )
