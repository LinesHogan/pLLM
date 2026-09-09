from enum import Enum
from contextlib import AbstractContextManager

import torch
from torch.nn import Module


class TaskType(Enum):
    CAUSAL_LM: TaskType


class LoraConfig:
    def __init__(
        self,
        *,
        task_type: TaskType,
        inference_mode: bool,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: list[str],
        bias: str,
    ) -> None: ...


class PeftModelForCausalLM(Module):
    def disable_adapter(self) -> AbstractContextManager[None]: ...
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        use_cache: bool,
        return_dict: bool,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> object: ...


def get_peft_model(
    model: Module,
    peft_config: LoraConfig,
) -> Module: ...
