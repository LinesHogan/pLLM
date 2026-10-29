from collections.abc import Callable
from typing import Generic, TypeVar

import torch

F = TypeVar("F")

class _Impl(Generic[F]):
    impl_fn: F

class _Op(Generic[F]):
    impls: dict[str, _Impl[F]]

rms_norm: _Op[Callable[[torch.Tensor, torch.Tensor, float, int | None], torch.Tensor]]
fused_add_rms_norm: _Op[Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float, int | None], tuple[torch.Tensor, torch.Tensor]]]
