"""vLLM Qwen2 components used by workload binding and qualified effects."""

import torch
from torch.nn import Module
from transformers import Qwen2Config
from typing_extensions import TypeVar
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models._decoder import DecoderAttention, DecoderLayer, DecoderMLP, DecoderModel

class _SiluAndMul(Module):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor: ...

class _RotaryEmbedding(Module):
    def forward_native(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Qwen2MLP(DecoderMLP):
    gate_up_proj: LinearBase
    act_fn: _SiluAndMul

class Qwen2Attention(DecoderAttention):
    o_proj: LinearBase
    rotary_emb: _RotaryEmbedding
    num_heads: int
    num_kv_heads: int
    head_dim: int

class Qwen2DecoderLayer(DecoderLayer[Qwen2Attention, Qwen2MLP]): ...

_Layer = TypeVar("_Layer", bound=Module, default=Qwen2DecoderLayer, covariant=True)

class Qwen2Model(DecoderModel[_Layer]): ...

class Qwen2ForCausalLM(Module):
    model: Qwen2Model
    lm_head: VocabParallelEmbedding
    config: Qwen2Config
    quant_config: object | None

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
