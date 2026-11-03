from typing import overload

import torch
from torch.nn import Module


class RMSNorm(Module):
    hidden_size: int
    variance_epsilon: float
    variance_size_override: int | None
    has_weight: bool
    pass_weight: bool
    pass_weight_add: bool
    weight: torch.Tensor

    def __init__(self, hidden_size: int, eps: float = 1e-6, var_hidden_size: int | None = None,
                 has_weight: bool = True, dtype: torch.dtype | None = None) -> None: ...
    @overload
    def forward(self, x: torch.Tensor, residual: None = None) -> torch.Tensor: ...
    @overload
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def forward_cuda(self, x: torch.Tensor, residual: None = None) -> torch.Tensor: ...
    @overload
    def forward_cuda(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

class GemmaRMSNorm(Module):
    """x * (1 + weight): the stored weight is an offset from unit gain."""

    variance_epsilon: float
    weight: torch.Tensor

    @overload
    def forward(self, x: torch.Tensor, residual: None = None) -> torch.Tensor: ...
    @overload
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
