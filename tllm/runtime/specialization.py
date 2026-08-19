"""Typed cold-path specialization facts supplied by runtime backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch


@dataclass(frozen=True, slots=True)
class RuntimeSpecializationSpec:
    rows: int
    hidden_size: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        if not isinstance(self.rows, int) or isinstance(self.rows, bool) or self.rows < 1:
            raise ValueError("RuntimeSpecializationSpec.rows must be a positive int")
        if (
            not isinstance(self.hidden_size, int)
            or isinstance(self.hidden_size, bool)
            or self.hidden_size < 1
        ):
            raise ValueError(
                "RuntimeSpecializationSpec.hidden_size must be a positive int"
            )
        if not isinstance(cast(object, self.dtype), torch.dtype):
            raise TypeError("RuntimeSpecializationSpec.dtype must be torch.dtype")
        if not isinstance(cast(object, self.device), torch.device):
            raise TypeError("RuntimeSpecializationSpec.device must be torch.device")


__all__ = ["RuntimeSpecializationSpec"]
