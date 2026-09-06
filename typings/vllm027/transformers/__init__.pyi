from __future__ import annotations

from os import PathLike
from typing import Self

import torch


class PreTrainedModel(torch.nn.Module):
    def to(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Self: ...


class AutoModelForCausalLM:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *,
        revision: str,
        local_files_only: bool,
        trust_remote_code: bool,
        torch_dtype: torch.dtype,
    ) -> PreTrainedModel: ...
