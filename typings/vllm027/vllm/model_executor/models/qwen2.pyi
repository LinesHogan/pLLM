from collections.abc import Callable, Sequence

import torch
from torch.nn import Module, Parameter
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import Attention


_LinearOutput = torch.Tensor | tuple[torch.Tensor, Parameter | None]


class Qwen2DownProjection(Module):
    weight: Parameter
    bias: Parameter | None
    tp_size: int
    return_bias: bool
    forward: Callable[[Qwen2DownProjection, torch.Tensor], _LinearOutput]


class Qwen2MLP(Module):
    down_proj: Qwen2DownProjection

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class Qwen2Attention(Module):
    attn: Attention
    qkv_proj: Qwen2DownProjection
    q_size: int
    kv_size: int

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor: ...


class Qwen2DecoderLayer(Module):
    mlp: Qwen2MLP
    self_attn: Qwen2Attention
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    forward: Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]]


class Qwen2Model(Module):
    layers: Sequence[Qwen2DecoderLayer]
    norm: RMSNorm
    forward: Callable[[torch.Tensor | None, torch.Tensor, IntermediateTensors | None, torch.Tensor | None], torch.Tensor | IntermediateTensors]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...


class Qwen2ModelConfig:
    vocab_size: int
    hidden_size: int


class Qwen2LMHead(Module):
    weight: Parameter


class Qwen2ForCausalLM(Module):
    model: Qwen2Model
    lm_head: Qwen2LMHead
    config: Qwen2ModelConfig
    quant_config: object | None

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
