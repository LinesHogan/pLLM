"""vLLM projection surface used by parameter overlays."""

from collections.abc import Callable

import torch
from torch.nn import Module, Parameter

_LinearOutput = torch.Tensor | tuple[torch.Tensor, Parameter | None]

class LinearBase(Module):
    weight: Parameter
    bias: Parameter | None
    tp_size: int
    return_bias: bool
    forward: Callable[[LinearBase, torch.Tensor], _LinearOutput]
