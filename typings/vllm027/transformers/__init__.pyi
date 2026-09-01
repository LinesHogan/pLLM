from __future__ import annotations

from os import PathLike

import torch


class PreTrainedModel(torch.nn.Module):
    def to(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PreTrainedModel: ...


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
