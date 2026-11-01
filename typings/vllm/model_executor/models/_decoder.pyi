"""Shared typing surface of the admitted two-stream vLLM decoders."""

from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

from torch import Tensor
from torch.nn import Module
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.sequence import IntermediateTensors

class SiluAndMul(Module):
    def forward_native(self, x: Tensor) -> Tensor: ...

class DecoderAttention(Module):
    o_proj: LinearBase
    rotary_emb: RotaryEmbedding
    num_heads: int
    num_kv_heads: int
    head_dim: int
    attn: Attention
    qkv_proj: LinearBase
    q_size: int
    kv_size: int

    def forward(self, positions: Tensor, hidden_states: Tensor) -> Tensor: ...

class DecoderMLP(Module):
    gate_up_proj: LinearBase
    act_fn: SiluAndMul
    down_proj: LinearBase

    def forward(self, x: Tensor) -> Tensor: ...

_Attention = TypeVar("_Attention", bound=DecoderAttention)
_MLP = TypeVar("_MLP", bound=DecoderMLP)
_Layer = TypeVar("_Layer", bound=Module, covariant=True)

class DecoderLayer(Module, Generic[_Attention, _MLP]):
    self_attn: _Attention
    mlp: _MLP
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    forward: Callable[[Tensor, Tensor, Tensor | None], tuple[Tensor, Tensor]]

class DecoderModel(Module, Generic[_Layer]):
    quant_config: object | None
    embed_tokens: VocabParallelEmbedding
    layers: Sequence[_Layer]
    norm: RMSNorm
    do_not_compile: bool
    forward: Callable[[Tensor | None, Tensor, IntermediateTensors | None, Tensor | None], Tensor | IntermediateTensors]

    def embed_input_ids(self, input_ids: Tensor) -> Tensor: ...
