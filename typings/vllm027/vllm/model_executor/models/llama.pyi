"""vLLM Llama decoder surface used by ordinary workload binding."""

from collections.abc import Callable, Sequence

from torch import Tensor
from torch.nn import Module
from transformers import LlamaConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

class LlamaDecoderLayer(Module):
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    forward: Callable[[Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]]

class LlamaModel(Module):
    quant_config: object | None
    embed_tokens: VocabParallelEmbedding
    layers: Sequence[LlamaDecoderLayer]
    norm: RMSNorm
    do_not_compile: bool

    def embed_input_ids(self, input_ids: Tensor) -> Tensor: ...

class LlamaForCausalLM(Module):
    model: LlamaModel
    lm_head: VocabParallelEmbedding
    config: LlamaConfig

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...
