"""Embedding and tied or untied output weights used by decoder model bindings."""

from torch import Tensor, dtype
from torch.nn import Module, Parameter

class UnquantizedEmbeddingMethod: ...

class VocabParallelEmbedding(Module):
    weight: Parameter
    num_embeddings_padded: int
    quant_method: object

    def __init__(
        self, num_embeddings: int, embedding_dim: int, params_dtype: dtype | None = None, *, disable_tp: bool = False
    ) -> None: ...
    def forward(self, input_: Tensor) -> Tensor: ...

class ParallelLMHead(VocabParallelEmbedding): ...

def pad_vocab_size(vocab_size: int, pad_to: int = 64) -> int: ...
