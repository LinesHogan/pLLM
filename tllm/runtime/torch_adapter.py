"""Quarantined decoding of external Torch runtime value names."""

from __future__ import annotations

from typing import Any, cast

import torch


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


__all__ = ["runtime_torch_values"]
