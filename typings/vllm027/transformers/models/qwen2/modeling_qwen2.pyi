from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import torch
from transformers import PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class Qwen2Projection(torch.nn.Module):
    weight: torch.nn.Parameter


class Qwen2Attention(torch.nn.Module):
    q_proj: Qwen2Projection
    v_proj: Qwen2Projection


class Qwen2DecoderLayer(torch.nn.Module):
    self_attn: Qwen2Attention


class Qwen2Model(torch.nn.Module):
    layers: Sequence[Qwen2DecoderLayer]


class Qwen2ForCausalLM(PreTrainedModel):
    model: Qwen2Model

    def __init__(self, config: Qwen2Config) -> None: ...

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *,
        revision: str,
        local_files_only: bool,
        trust_remote_code: bool,
        torch_dtype: torch.dtype,
    ) -> Qwen2ForCausalLM: ...
