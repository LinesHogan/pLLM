#!/usr/bin/env python3
"""Runtime-owned device tensor lease contracts for advanced consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Protocol, runtime_checkable

import torch

from tllm.ports.base import BufferOwnership

LeaseLifetime = Literal["consume_call", "step"]


@runtime_checkable
class DeviceTensorWriteSink(Protocol):
    """Address-stable runtime output slot exposed to a consumer."""

    @property
    def max_rows(self) -> int: ...

    def stage(
        self,
        *,
        delta: torch.Tensor,
        row_idx: torch.Tensor,
        validate_rows: bool = False,
    ) -> None: ...


def _empty_write_outputs() -> Mapping[str, DeviceTensorWriteSink]:
    return {}


@dataclass(frozen=True)
class DeviceTensorLease:
    entries: Mapping[str, torch.Tensor]
    active_rows: int
    ownership: BufferOwnership = BufferOwnership.RUNTIME_LEASE
    lifetime: LeaseLifetime = "consume_call"
    outputs: Mapping[str, DeviceTensorWriteSink] = field(
        default_factory=_empty_write_outputs
    )

    def __post_init__(self) -> None:
        if int(self.active_rows) < 0:
            raise ValueError("DeviceTensorLease active_rows must be non-negative")
        if str(self.ownership) != "runtime_lease":
            raise ValueError("DeviceTensorLease currently requires ownership='runtime_lease'")
        if str(self.lifetime) not in {"consume_call", "step"}:
            raise ValueError("DeviceTensorLease lifetime must be 'consume_call' or 'step'")
        for name, tensor in self.entries.items():
            if not isinstance(tensor, torch.Tensor):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
                raise ValueError(f"DeviceTensorLease entry `{name}` must be a torch.Tensor")
            if tensor.ndim < 1:
                raise ValueError(f"DeviceTensorLease entry `{name}` must have a row dimension")
            if int(self.active_rows) > int(tensor.shape[0]):
                raise ValueError(f"DeviceTensorLease active_rows exceeds entry `{name}` row capacity")
        for name, sink in self.outputs.items():
            if not isinstance(sink, DeviceTensorWriteSink):  # pyright: ignore[reportUnnecessaryIsInstance]  # boundary guard for untyped callers
                raise ValueError(f"DeviceTensorLease output `{name}` must provide a typed stage method")

    def output(self, role: str) -> DeviceTensorWriteSink:
        """Return a declared output by role with one checked access point."""
        try:
            return self.outputs[str(role)]
        except KeyError as exc:
            available = ", ".join(sorted(self.outputs)) or "<none>"
            raise KeyError(f"unknown device write output {role!r}; available: {available}") from exc
