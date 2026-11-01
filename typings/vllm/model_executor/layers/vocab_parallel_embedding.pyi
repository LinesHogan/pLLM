"""Embedding and tied or untied output weights used by decoder model bindings."""

from torch import Tensor
from torch.nn import Module, Parameter

class VocabParallelEmbedding(Module):
    weight: Parameter
    num_embeddings_padded: int

    def forward(self, input_: Tensor) -> Tensor: ...

class ParallelLMHead(VocabParallelEmbedding): ...
