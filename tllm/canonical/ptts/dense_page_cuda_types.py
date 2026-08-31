"""Opaque handles and typed payloads for canonical CUDA dense pages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.contracts.state_ref import StateRef


@dataclass(frozen=True, slots=True, weakref_slot=True)
class CudaDensePageVersionPin:
    pin_id: int
    refs: tuple[StateRef, ...]
    stream: torch.cuda.Stream


@dataclass(frozen=True, slots=True)
class CudaDensePageInitialRoot:
    stream: torch.cuda.Stream
    tensors: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if (
            type(self.stream) is not torch.cuda.Stream
            or type(self.tensors) is not tuple
            or not self.tensors
            or any(type(tensor) is not torch.Tensor for tensor in self.tensors)
        ):
            raise TypeError("CUDA initial root requires exact stream-owned tensors")


@dataclass(frozen=True, slots=True)
class CudaDensePageAdmission:
    ref: StateRef
    initial: CudaDensePageInitialRoot

    def __post_init__(self) -> None:
        if type(self.ref) is not StateRef or type(
            self.initial
        ) is not CudaDensePageInitialRoot:
            raise TypeError("CUDA admission requires a StateRef and initial root")


@dataclass(frozen=True, slots=True)
class CudaDensePagePreparedAdmission:
    transaction_id: int

    def __post_init__(self) -> None:
        if type(self.transaction_id) is not int or self.transaction_id < 1:
            raise ValueError("CUDA admission token requires a transaction ID")


@dataclass(frozen=True, slots=True)
class CudaDensePageTransition:
    transaction_id: int

    def __post_init__(self) -> None:
        if type(self.transaction_id) is not int or self.transaction_id < 1:
            raise ValueError("CUDA transition token requires a transaction ID")


@dataclass(frozen=True, slots=True)
class CudaDensePagePublication:
    refs: tuple[StateRef, ...]

    def __post_init__(self) -> None:
        if (
            not self.refs
            or len(self.refs) != len(set(self.refs))
        ):
            raise ValueError("CUDA publication receipt is invalid")


@dataclass(frozen=True, slots=True, weakref_slot=True)
class CudaDensePageRelease:
    release_id: int
    pin: CudaDensePageVersionPin

    def __post_init__(self) -> None:
        if (
            type(self.release_id) is not int
            or type(self.pin) is not CudaDensePageVersionPin
            or self.release_id < 1
        ):
            raise ValueError("CUDA release receipt is invalid")


@dataclass(frozen=True, slots=True)
class CudaDensePageGradientBatch:
    batch_id: int

    def __post_init__(self) -> None:
        if type(self.batch_id) is not int or self.batch_id < 1:
            raise ValueError("CUDA gradient token requires a batch ID")


__all__ = [
    "CudaDensePageAdmission",
    "CudaDensePageGradientBatch",
    "CudaDensePageInitialRoot",
    "CudaDensePagePreparedAdmission",
    "CudaDensePagePublication",
    "CudaDensePageRelease",
    "CudaDensePageTransition",
    "CudaDensePageVersionPin",
]
