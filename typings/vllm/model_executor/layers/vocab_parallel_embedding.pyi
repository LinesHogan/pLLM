"""Embedding and tied or untied output weights used by decoder model bindings."""

from torch import Tensor
from torch.nn import Module, Parameter

class UnquantizedEmbeddingMethod: ...

class VocabParallelEmbedding(Module):
    weight: Parameter
    num_embeddings_padded: int
    quant_method: object

    def forward(self, input_: Tensor) -> Tensor: ...

class ParallelLMHead(VocabParallelEmbedding): ...

def pad_vocab_size(vocab_size: int, pad_to: int = 64) -> int: ...
