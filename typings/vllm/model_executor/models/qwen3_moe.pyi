"""vLLM Qwen3-MoE decoder surface: Qwen3 attention with routed-expert (or dense) MLP layers."""

from torch import Tensor
from torch.nn import Module
from transformers import Qwen3MoeConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models._decoder import DecoderAttention, DecoderLayer, DecoderMLP, DecoderModel

class Qwen3MoeAttention(DecoderAttention):
    q_norm: RMSNorm
    k_norm: RMSNorm

class Qwen3MoeMLP(DecoderMLP): ...

class Qwen3MoeSparseMoeBlock(Module):
    def forward(self, hidden_states: Tensor) -> Tensor: ...

class Qwen3MoeDecoderLayer(DecoderLayer[Qwen3MoeAttention, Qwen3MoeMLP | Qwen3MoeSparseMoeBlock]): ...
class Qwen3MoeModel(DecoderModel[Qwen3MoeDecoderLayer]): ...

class Qwen3MoeForCausalLM(Module):
    model: Qwen3MoeModel
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: Qwen3MoeConfig

    def compute_logits(self, hidden_states: Tensor) -> Tensor | None: ...
