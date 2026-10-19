"""Meta execution and storage liveness used by cold workload resource derivation.

Storage weak references follow views and autograd's saved tensors, including references
held by Python containers. Inputs predate the trace and do not count as workspace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Protocol, cast

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode

from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw


def allocation_bytes(size: int) -> int:
    """CUDA allocator's minimum block size, also used for conservative CPU facts."""
    return (size + 511) // 512 * 512


def tensors(value: object) -> tuple[torch.Tensor, ...]:
    """Flatten tensor containers without retaining their owners."""
    if isinstance(value, torch.Tensor):
        return (value,)
    if is_dataclass(value) and not isinstance(value, type):
        return tensors(tuple(getattr(value, field.name) for field in fields(value)))
    if isinstance(value, Mapping):
        return tensors(tuple(cast(Mapping[object, object], value).values()))
    if isinstance(value, (tuple, list)):
        return tuple(tensor for item in cast(tuple[object, ...], value) for tensor in tensors(item))
    return ()


class _StorageReference(Protocol):
    cdata: int


class MemoryTrace(TorchDispatchMode):
    """Measure live allocations while executing shape-only ATen mathematics."""

    def __init__(self) -> None:
        # Torch leaves the optional dispatch-key parameter untyped.
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.storages: dict[int, tuple[StorageWeakRef, int]] = {}
        self.peak = 0

    def _remember(self, values: object, *, owned: bool) -> None:
        for tensor in tensors(values):
            storage = tensor.untyped_storage()
            reference = StorageWeakRef(storage)
            key = cast(_StorageReference, reference).cdata
            if key not in self.storages:
                self.storages[key] = (reference, allocation_bytes(storage.nbytes()) if owned else 0)

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        self.storages = {key: value for key, value in self.storages.items() if not value[0].expired()}
        self._remember((args, kwargs or {}), owned=False)
        result = func(*args, **(kwargs or {}))
        self._remember(result, owned=True)
        self.peak = max(self.peak, sum(size for _, size in self.storages.values()))
        return result


@dataclass(frozen=True)
class MemoryUsage:
    """Peak task scratch and storage retained by its returned tensors."""

    workspace: int
    output: int


def measure(function: Callable[[], object]) -> MemoryUsage:
    """Run a meta-only tensor program, including its declared backward and optimizer."""
    with torch.inference_mode(False), MemoryTrace() as trace:
        result = function()
        output = sum(allocation_bytes(value.numel() * value.element_size()) for value in tensors(result))
    return MemoryUsage(trace.peak, output)


def optimizer_scratch(parameters: tuple[torch.Tensor, ...], config: AdamWConfig) -> MemoryUsage:
    """Trace the actual CUDA foreach optimizer on fake operands, without initializing CUDA."""
    mode = FakeTensorMode()
    with mode:

        def fake(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
            return FakeTensor(mode, torch.empty(shape, dtype=dtype, device="meta"), torch.device("cuda:0"))

        roots = tuple(
            tuple(fake((), torch.float32) if i == 2 else fake(tuple(value.shape), value.dtype) for i in range(5))
            for value in parameters
        )
        gradients = tuple(fake(tuple(value.shape), value.dtype) for value in parameters)
        return measure(
            lambda: apply_adamw(roots, AdamWFields(*(LayoutFieldId(i) for i in range(5))), config, gradients)
        )
