"""Storage liveness while the planner executes tensor operations on shape-only operands."""

import weakref
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from pllm.kernels.adamw import AdamWConfig, AdamWFields, apply_adamw


@dataclass(frozen=True)
class MemoryUsage:
    """Task peak and allocations retained after its return, in allocator-rounded bytes."""

    workspace: int
    output: int


class MemoryTrace(TorchDispatchMode):
    """Follow storage lifetimes, including aliases and autograd's saved tensors."""

    active: ContextVar["MemoryTrace | None"] = ContextVar("memory_trace", default=None)

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.sizes: dict[StorageWeakRef, int] = {}
        self.live = 0
        self.peak = 0

    def _forget(self, ref: StorageWeakRef) -> None:
        self.live -= self.sizes.pop(ref)

    def _remember(self, values: object, owned: bool) -> None:
        for value in tree_leaves(values):
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage()
                ref = StorageWeakRef(storage)
                if ref in self.sizes:
                    continue
                size = (storage.nbytes() + 511) // 512 * 512
                # Native large-pool reuse can keep an unsplit tail of up to 1 MiB.
                size += (1 << 20) if size > (1 << 20) else 0
                self.sizes[ref] = size if owned else 0
                self.live += self.sizes[ref]
                # Torch keeps a storage's Python object as long as its bytes, so this runs exactly at the free;
                # keeping the total incremental makes tracing linear in the number of operations.
                weakref.finalize(storage, self._forget, ref)

    def __torch_dispatch__(
        self,
        func: Callable[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        self._remember((args, kwargs), False)
        result = func(*args, **(kwargs or {}))
        self._remember(result, True)
        self.peak = max(self.peak, self.live)
        return result

    @classmethod
    def reserve(cls, size: int) -> None:
        """Include an executable's opaque workspace at the call's live frontier."""
        trace = cls.active.get()
        if trace is not None:
            trace.peak = max(trace.peak, trace.live + size)


def measure(function: Callable[[], object]) -> MemoryUsage:
    """Trace the actual callable; keep its return alive while measuring retained storage."""
    with torch.inference_mode(False), MemoryTrace() as trace:
        token = MemoryTrace.active.set(trace)
        try:
            _result = function()
            return MemoryUsage(trace.peak, trace.live)
        finally:
            MemoryTrace.active.reset(token)


def apply_adamw_or_reserve(
    roots: Sequence[tuple[torch.Tensor, ...]],
    fields: AdamWFields,
    config: AdamWConfig,
    gradients: tuple[torch.Tensor, ...],
) -> None:
    """Apply AdamW, or, on meta pages while planning, reserve the scratch the CUDA kernel would allocate."""
    if not (roots and roots[0][0].is_meta):
        apply_adamw(roots, fields, config, gradients)
        return
    # Torch's capturable implementation requires a CUDA tag; the storage stays meta.
    with FakeTensorMode() as mode:

        def operand(value: torch.Tensor) -> torch.Tensor:
            return FakeTensor(mode, value, torch.device("cuda:0"))

        pages = tuple(tuple(operand(value) for value in root) for root in roots)
        values = tuple(operand(value) for value in gradients)
        scratch = measure(lambda: apply_adamw(pages, fields, config, values))
    MemoryTrace.reserve(scratch.workspace)
