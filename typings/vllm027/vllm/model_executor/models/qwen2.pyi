"""vLLM Qwen2 components used by workload binding and qualified effects."""

from collections.abc import Callable, Sequence
from typing import Generic

import torch
from torch.nn import Module, Parameter
from transformers import Qwen2Config
from typing_extensions import TypeVar
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.sequence import IntermediateTensors

_LinearOutput = torch.Tensor | tuple[torch.Tensor, Parameter | None]

class Qwen2DownProjection(Module):
    weight: Parameter
    bias: Parameter | None
    tp_size: int
    return_bias: bool
    forward: Callable[[Qwen2DownProjection, torch.Tensor], _LinearOutput]

class _SiluAndMul(Module):
    def forward_native(self, x: torch.Tensor) -> torch.Tensor: ...

class _RotaryEmbedding(Module):
    def forward_native(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

class Qwen2MLP(Module):
    gate_up_proj: Qwen2DownProjection
    act_fn: _SiluAndMul
    down_proj: Qwen2DownProjection

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Qwen2Attention(Module):
    attn: Attention
    qkv_proj: Qwen2DownProjection
    o_proj: Qwen2DownProjection
    rotary_emb: _RotaryEmbedding
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_size: int
    kv_size: int

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor: ...

class Qwen2DecoderLayer(Module):
    mlp: Qwen2MLP
    self_attn: Qwen2Attention
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]

_Layer = TypeVar("_Layer", bound=Module, default=Qwen2DecoderLayer, covariant=True)

class Qwen2Model(Module, Generic[_Layer]):
    quant_config: object | None
    embed_tokens: VocabParallelEmbedding
    do_not_compile: bool
    layers: Sequence[_Layer]
    norm: RMSNorm
    forward: Callable[
        [torch.Tensor | None, torch.Tensor, IntermediateTensors | None, torch.Tensor | None],
        torch.Tensor | IntermediateTensors,
    ]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...

class Qwen2ForCausalLM(Module):
    model: Qwen2Model
    lm_head: VocabParallelEmbedding
    config: Qwen2Config
    quant_config: object | None

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
