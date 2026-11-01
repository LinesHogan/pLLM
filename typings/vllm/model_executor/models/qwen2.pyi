"""vLLM Qwen2 components used by workload binding and qualified effects."""

import torch
from torch.nn import Module
from transformers import Qwen2Config
from typing_extensions import TypeVar
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models._decoder import DecoderAttention, DecoderLayer, DecoderMLP, DecoderModel

class Qwen2MLP(DecoderMLP): ...

class Qwen2Attention(DecoderAttention):
    qk_norm: bool
    q_norm: RMSNorm
    k_norm: RMSNorm

class Qwen2DecoderLayer(DecoderLayer[Qwen2Attention, Qwen2MLP]): ...

_Layer = TypeVar("_Layer", bound=Module, default=Qwen2DecoderLayer, covariant=True)

class Qwen2Model(DecoderModel[_Layer]): ...

class Qwen2ForCausalLM(Module):
    model: Qwen2Model
    logits_processor: LogitsProcessor
    lm_head: VocabParallelEmbedding
    config: Qwen2Config
    quant_config: object | None

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
