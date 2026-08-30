"""Quarantined decoding of external Torch runtime value names."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch.utils import _pytree


def torch_assert_async(condition: torch.Tensor, message: str) -> None:
    """Invoke Torch's compiler-aware device assertion in one quarantine."""

    dynamic = cast(Any, torch)
    dynamic._assert_async(condition, message)


def torch_record_stream_owners(
    values: tuple[object, ...], stream: object
) -> None:
    """Transfer opaque Tensor storage lifetime to one CUDA stream."""

    if not isinstance(stream, torch.cuda.Stream):
        return
    leaves = cast(list[object], _pytree.tree_leaves(values))
    for value in leaves:
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            value.record_stream(stream)


def runtime_torch_values(
    device_wire: str, dtype_wire: str
) -> tuple[torch.device, torch.dtype]:
    if not dtype_wire.startswith("torch."):
        raise RuntimeError("worker returned an invalid Torch dtype specialization")
    dtype = cast(Any, getattr(torch, dtype_wire.removeprefix("torch."), None))
    if not isinstance(dtype, torch.dtype) or str(dtype) != dtype_wire:
        raise RuntimeError("worker returned an invalid Torch dtype specialization")
    try:
        device = torch.device(device_wire)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("worker returned an invalid Torch device specialization") from exc
    if str(device) != device_wire:
        raise RuntimeError("worker returned a noncanonical Torch device specialization")
    return device, dtype


__all__ = [
    "runtime_torch_values",
    "torch_record_stream_owners",
    "torch_assert_async",
]
