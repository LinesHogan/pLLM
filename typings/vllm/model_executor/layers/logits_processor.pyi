"""Inference logits scaling used by causal training."""

from torch.nn import Module

class LogitsProcessor(Module):
    scale: float
