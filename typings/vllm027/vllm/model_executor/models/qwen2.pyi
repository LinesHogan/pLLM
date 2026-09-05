from collections.abc import Callable, Sequence

import torch
from torch.nn import Module, Parameter


_LinearOutput = torch.Tensor | tuple[torch.Tensor, Parameter | None]


class Qwen2DownProjection(Module):
    weight: Parameter
    bias: Parameter | None
    tp_size: int
    return_bias: bool
    forward: Callable[[Qwen2DownProjection, torch.Tensor], _LinearOutput]


class Qwen2MLP(Module):
    down_proj: Qwen2DownProjection
    forward: Callable[[Qwen2MLP, torch.Tensor], torch.Tensor]


class Qwen2DecoderLayer(Module):
    mlp: Qwen2MLP


class Qwen2Model(Module):
    layers: Sequence[Qwen2DecoderLayer]


class Qwen2ModelConfig:
    hidden_size: int


class Qwen2LMHead(Module):
    weight: Parameter


class Qwen2ForCausalLM(Module):
    model: Qwen2Model
    lm_head: Qwen2LMHead
    config: Qwen2ModelConfig
    quant_config: object | None

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
