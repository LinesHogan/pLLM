"""Torch private-API shims: tracing decompositions, checkpoints, CUDA storage and allocator facts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypedDict
from uuid import UUID

import torch
from torch import _decomp
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor


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
