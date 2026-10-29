"""vLLM Qwen3 decoder surface used by ordinary workload binding."""

from collections.abc import Callable

from torch import Tensor
from torch.nn import Module
from transformers import Qwen3Config
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen2 import Qwen2MLP, Qwen2Model

class Qwen3Attention(Module):
    q_norm: RMSNorm
    k_norm: RMSNorm
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_size: int
    kv_size: int

    def forward(self, positions: Tensor, hidden_states: Tensor) -> Tensor: ...

class Qwen3DecoderLayer(Module):
    self_attn: Qwen3Attention
    mlp: Qwen2MLP
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    forward: Callable[[Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]]

class Qwen3Model(Qwen2Model[Qwen3DecoderLayer]): ...

class Qwen3ForCausalLM(Module):
    model: Qwen3Model
    lm_head: VocabParallelEmbedding
    config: Qwen3Config
    quant_config: object | None

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...
