"""Shape-only scan and optimizer planning must work in a CUDA-free worker."""

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_meta_scan_and_adamw_do_not_initialize_cuda():
    script = """
from unittest.mock import patch
import torch
assert not torch.cuda.is_initialized()
def forbidden(*args, **kwargs):
    raise AssertionError("planning must not allocate tensors or initialize CUDA")
from pllm.execution.torch_private import measure
from pllm import scan
from pllm.execution.parameter_projection import prepare_projection_read
from pllm.contracts.parameter_overlay import ParameterOverlayOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from pllm.execution.memory import apply_adamw_or_reserve
from pllm.kernels.adamw import AdamWConfig, AdamWFields
with patch.object(torch.cuda, '_lazy_init', forbidden):
    roots = ((torch.empty(32, device='meta'),) * 2 + (torch.empty((), device='meta'),) + (torch.empty(32, device='meta'),) * 2,)
    measure(lambda: apply_adamw_or_reserve(roots, AdamWFields(0, 1, 2, 3, 4), AdamWConfig(.01), (torch.empty(32, device='meta'),)))
    scan(lambda carry, x: (carry + x.sum(0), x + carry), carry=torch.empty(3, device="meta"), over=(torch.empty(4, 3, device="meta"),), chunk=2)
    with FakeTensorMode():
        examples = (
            torch.empty((8, 8), device="cuda"),
            torch.empty((2, 1), device="cuda", dtype=torch.int64),
            torch.empty((2, 4), device="cuda", dtype=torch.int64),
            torch.empty((8,), device="cuda", dtype=torch.int64),
            torch.empty((32, 2, 4, 4), device="cuda"),
        )
        prepare_projection_read(examples, ParameterOverlayOperator.HEAD_ADD, 128 << 20)
assert not torch.cuda.is_initialized()
"""
    subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=45)
