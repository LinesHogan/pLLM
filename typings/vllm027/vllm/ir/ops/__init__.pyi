from contextlib import AbstractContextManager

import torch
from vllm.ir.op import IrOpImpl


class _FusedAddRmsNorm:
    def dispatch(self, x: torch.Tensor, x_residual: torch.Tensor, weight: torch.Tensor | None,
                 epsilon: float, variance_size: int | None = None) -> IrOpImpl: ...
    def get_priority(self) -> list[str]: ...
    def set_priority(self, priority: list[str]) -> AbstractContextManager[None]: ...


fused_add_rms_norm: _FusedAddRmsNorm
