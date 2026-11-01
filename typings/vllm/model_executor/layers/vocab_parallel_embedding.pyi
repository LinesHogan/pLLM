"""Embedding and tied or untied output weights used by decoder model bindings."""

from torch import Tensor
from torch.nn import Module, Parameter

class VocabParallelEmbedding(Module):
    weight: Parameter

    def forward(self, input_: Tensor) -> Tensor: ...

class ParallelLMHead(VocabParallelEmbedding): ...
