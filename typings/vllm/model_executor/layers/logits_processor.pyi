"""Inference logits scaling used by causal training."""

from torch.nn import Module
from torch import dtype

class LogitsProcessor(Module):
    scale: float
    soft_cap: float | None
    logits_as_input: bool
    head_dtype: dtype | None
