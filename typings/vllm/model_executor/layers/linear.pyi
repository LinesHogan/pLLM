"""vLLM projection surface used by parameter overlays and causal training.

Quantized projections pack their weight (and may name it otherwise); the per-partition sizes and
params_dtype describe the dense [output, input] matrix the kernel applies.
"""

from collections.abc import Callable

import torch
from torch.nn import Module, Parameter

_LinearOutput = torch.Tensor | tuple[torch.Tensor, Parameter | None]

class LinearMethodBase:
    def apply(self, layer: LinearBase, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor: ...

class LinearBase(Module):
    weight: Parameter
    bias: Parameter | None
    input_size_per_partition: int
    output_size_per_partition: int
    params_dtype: torch.dtype
    quant_method: LinearMethodBase
    tp_size: int
    return_bias: bool
    # Replaceable per instance: overlays install their own bound forward.
    forward: Callable[[torch.Tensor], _LinearOutput]
