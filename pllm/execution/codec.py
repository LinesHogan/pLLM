"""Shared wire-codec access and Torch storage, profiling and route-tracing adapters."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from uuid import UUID

import torch
from torch import _decomp
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor

from pllm.contracts.codec import (
    boundary_digest as boundary_digest,
    decode_boundary as decode_boundary,
    encode_boundary as encode_boundary,
)


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


@dataclass(frozen=True, slots=True)
class CudaProfileTotals:
    kernel_count: int
    kernel_microseconds: float
    cuda_api_microseconds: float
    cuda_enqueue_api_microseconds: float
    cuda_wait_api_microseconds: float
    gpu_activity_microseconds: float
    memcpy_bytes: int
    cuda_launch_api_count: int
    cuda_graph_launch_api_count: int


class _TraceArgs(TypedDict):
    bytes: int


class _TraceEvent(TypedDict):
    ph: str
    cat: str
    name: str
    pid: int | str
    tid: int | str
    ts: float
    dur: float
    args: _TraceArgs


def _span_totals(spans: dict[tuple[str, str], list[tuple[float, float]]]) -> tuple[int, float]:
    count = 0
    total = 0.0
    for intervals in spans.values():
        end = float("-inf")
        for start, stop in sorted(intervals):
            count += start >= end
            total += max(0.0, stop - max(start, end))
            end = max(end, stop)
    return count, total


def cuda_profile_totals(path: Path) -> CudaProfileTotals:
    """Decode Kineto activity, merging nested API spans per thread and GPU spans per device.

    Enqueue API time is not total Python submission time and calls may block internally.
    Launch counts merge nested runtime/driver calls on each host thread; graph
    launches are a subset, not the number of device kernels executed by a graph.
    Kernel duration sums may overlap; GPU activity includes memcpy/memset without overlap.
    Memcpy bytes exclude kernel-internal loads/stores and all allocator figures.
    Foreign trace metadata is deliberately not a closed PLLM wire contract.
    """
    events: list[_TraceEvent] = json.loads(path.read_text())["traceEvents"]
    kernels, kernel_us, transfer = 0, 0.0, 0
    spans: dict[str, dict[tuple[str, str], list[tuple[float, float]]]] = {
        kind: {} for kind in ("api", "enqueue", "wait", "gpu", "launch", "graph")
    }
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in (
            "kernel",
            "cuda_runtime",
            "cuda_driver",
            "gpu_memcpy",
            "gpu_memset",
        ):
            continue
        duration = float(event["dur"])
        start = float(event["ts"])
        device_event = category in ("kernel", "gpu_memcpy", "gpu_memset")
        key = (str(event["pid"]), "device" if device_event else str(event["tid"]))
        interval = (start, start + duration)
        spans["gpu" if device_event else "api"].setdefault(key, []).append(interval)
        if category == "kernel":
            kernels += 1
            kernel_us += duration
        elif category == "gpu_memcpy":
            transfer += int(event["args"]["bytes"])
        elif not device_event:
            name = str(event["name"]).lower()
            if "synchronize" in name:
                spans["wait"].setdefault(key, []).append(interval)
            elif "launch" in name or "async" in name and ("memcpy" in name or "memset" in name):
                spans["enqueue"].setdefault(key, []).append(interval)
                if "launch" in name:
                    spans["launch"].setdefault(key, []).append(interval)
                    if "graphlaunch" in name:
                        spans["graph"].setdefault(key, []).append(interval)
    totals = {key: _span_totals(value) for key, value in spans.items()}
    return CudaProfileTotals(
        kernels,
        kernel_us,
        totals["api"][1],
        totals["enqueue"][1],
        totals["wait"][1],
        totals["gpu"][1],
        transfer,
        totals["launch"][0],
        totals["graph"][0],
    )
