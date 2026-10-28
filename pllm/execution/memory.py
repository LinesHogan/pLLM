"""AdamW execution and its fake-tensor workspace reservation for task planning."""

from collections.abc import Sequence

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from pllm.execution.torch_private import MemoryTrace, measure
from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw


def apply_adamw_or_reserve(
    roots: Sequence[tuple[torch.Tensor, ...]],
    fields: AdamWFields,
    config: AdamWConfig,
    gradients: tuple[torch.Tensor, ...],
) -> None:
    """Apply AdamW, or, on meta pages while planning, reserve the scratch the CUDA kernel would allocate."""
    if not (roots and roots[0][0].is_meta):
        apply_adamw(roots, fields, config, gradients)
        return
    # Torch's capturable implementation requires a CUDA tag; the storage stays meta.
    with FakeTensorMode() as mode:

        def operand(value: torch.Tensor) -> torch.Tensor:
            return FakeTensor(mode, value, torch.device("cuda:0"))

        pages = tuple(tuple(operand(value) for value in root) for root in roots)
        values = tuple(operand(value) for value in gradients)
        scratch = measure(lambda: apply_adamw(pages, fields, config, values))
    MemoryTrace.reserve(scratch.workspace)
