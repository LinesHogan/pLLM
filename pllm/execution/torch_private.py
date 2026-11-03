"""Typed Torch API boundary for execution: tracing, module initialization, storage and memory accounting."""

from __future__ import annotations

import weakref
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from uuid import UUID

import torch
from torch import _decomp
from torch._functorch import compile_utils
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _python_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

fork_rng: Callable[[list[int]], AbstractContextManager[None]] = torch.random.fork_rng  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
manual_seed: Callable[[int], torch.Generator] = torch.manual_seed  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
fx_graph_cse: Callable[[torch.fx.Graph], torch.fx.Graph] = compile_utils.fx_graph_cse  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

CudaPoolHandle = torch.cuda._POOL_HANDLE  # pyright: ignore[reportPrivateUsage]
disable_current_modes = _python_dispatch._disable_current_modes  # pyright: ignore[reportPrivateUsage]


def register_gradient_hook(tensor: torch.Tensor, hook: Callable[[torch.Tensor], torch.Tensor]) -> None:
    tensor.register_hook(hook)  # pyright: ignore[reportUnknownMemberType]


def tensor_decompositions(operators: tuple[OpOverload, ...]) -> dict[OpOverload, Callable[..., torch.Tensor]]:
    """Torch decompositions for fake tracing without AOT's CUDA RNG initialization."""
    return {  # pyright: ignore[reportReturnType,reportUnknownVariableType]
        **_decomp.core_aten_decompositions(),  # pyright: ignore[reportUnknownMemberType]
        **_decomp.get_decompositions(operators),  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    }


def check_functional_aten(target: object) -> None:
    if (
        not isinstance(target, OpOverload)
        or target.namespace != "aten"
        or target._schema.is_mutable  # pyright: ignore[reportPrivateUsage]
        or torch.Tag.nondeterministic_seeded in target.tags
    ):
        raise ValueError(f"export requires deterministic functional ATen operators, got {target}")


def tensor_node_metadata(node: torch.fx.Node) -> FakeTensor | None:
    value: object = node.meta.get("val")
    return value if isinstance(value, FakeTensor) else None


def load_tensor_fields(path: Path, fields: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    """Decode a weights-only checkpoint into the caller's explicit tensor field order."""
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    tensors = (
        tuple(t for f in fields if isinstance(t := payload.get(f), torch.Tensor))  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(payload, dict)
        else ()
    )
    if len(tensors) != len(fields):
        raise ValueError(f"checkpoint {path.name} must map {fields} to tensors")
    return tensors


def cuda_device_uuid(device: torch.device) -> str:
    return str(UUID(str(torch.cuda.get_device_properties(device).uuid)))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]


CudaStorageHandle = tuple[int, bytes, int, int, bytes, int, bytes, bool]


def share_cuda_storage(tensor: torch.Tensor) -> CudaStorageHandle:
    return tensor.untyped_storage()._share_cuda_()  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]


def open_cuda_storage(handle: CudaStorageHandle) -> torch.UntypedStorage:
    return torch.UntypedStorage._new_shared_cuda(*handle)  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType]


class _PoolSegment(TypedDict):
    device: int
    total_size: int


def cuda_graph_pool_bytes(pool: tuple[int, int], device: int) -> int:
    """Decode PyTorch allocator facts at the foreign-data boundary, only in cold planning."""
    assert torch.cuda.get_allocator_backend() == "native" and pool != (0, 0)
    segments: list[_PoolSegment] = torch.cuda.memory_snapshot(pool, False)  # pyright: ignore[reportUnknownMemberType,reportAssignmentType]
    return sum(s["total_size"] for s in segments if s["device"] == device)


@dataclass(frozen=True)
class MemoryUsage:
    """Task peak, retained allocations and conservative capture reservation, in bytes."""

    workspace: int
    output: int
    capture: int = 0


class MemoryTrace(TorchDispatchMode):
    """Follow storage lifetimes, including aliases and autograd's saved tensors."""

    active: ContextVar["MemoryTrace | None"] = ContextVar("memory_trace", default=None)

    def __init__(self) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.sizes: dict[StorageWeakRef, int] = {}
        self.live = 0
        self.peak = 0
        self.small_allocations = 0
        self.large_allocations = 0
        self.oversize_allocations = 0

    def _capture_allocation(self, size: int) -> None:
        # Bound a private native allocator pool without assuming any reuse. Small requests use
        # 2 MiB segments, medium requests 20 MiB; each fills at least half a segment before
        # another is needed. Oversize requests get individually rounded 2 MiB segments.
        if size <= 1 << 20:
            self.small_allocations += size
        elif size < 10 << 20:
            self.large_allocations += size
        else:
            self.oversize_allocations += (size + (2 << 20) - 1) // (2 << 20) * (2 << 20)

    @property
    def capture_bytes(self) -> int:
        """Conservative no-reuse reservation, including each used pool's unfinished segment."""
        return (
            (2 * self.small_allocations + (2 << 20) if self.small_allocations else 0)
            + (2 * self.large_allocations + (20 << 20) if self.large_allocations else 0)
            + self.oversize_allocations
        )

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
                if owned:
                    self._capture_allocation(size)
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
            trace._capture_allocation(size)


def blas_workspace_bytes() -> int:
    """Read the current CUDA device's configured BLAS scratch at GPU planning boundaries."""
    return torch.backends.cuda.cublas_workspace_size() + torch.backends.cuda.cublaslt_workspace_size()


def measure(function: Callable[[], object]) -> MemoryUsage:
    """Trace the actual callable; keep its return alive while measuring retained storage."""
    with torch.inference_mode(False), MemoryTrace() as trace:
        token = MemoryTrace.active.set(trace)
        try:
            _result = function()
            return MemoryUsage(trace.peak, trace.live, trace.capture_bytes)
        finally:
            MemoryTrace.active.reset(token)
