"""Phi-3 long-context rotary positional embedding stub."""

from torch import Tensor
from torch.nn import Module


class Phi3LongRoPEScaledRotaryEmbedding(Module):
    def __call__(self, positions: Tensor, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]: ...
