"""Shared wire-codec access and Torch storage, profiling and route-tracing adapters."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID

import torch
from torch import _decomp
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor

from pllm.contracts.codec import (
    boundary_digest as boundary_digest,
    boundary_scalar as _primitive,
    decode_boundary as decode_boundary,
    encode_boundary as encode_boundary,
)


class _TensorHooks(Protocol):
    def register_hook(self, hook: Callable[[torch.Tensor], torch.Tensor]) -> torch.utils.hooks.RemovableHandle: ...


def register_gradient_hook(tensor: torch.Tensor, hook: Callable[[torch.Tensor], torch.Tensor]) -> None:
    cast(_TensorHooks, tensor).register_hook(hook)


class _TensorDecompositions(Protocol):
    def get_decompositions(self, aten_ops: tuple[OpOverload, ...]) -> dict[object, object]: ...


def tensor_decompositions(operators: tuple[OpOverload, ...]) -> dict[OpOverload, Callable[..., torch.Tensor]]:
    """Torch decompositions for fake tracing without AOT's CUDA RNG initialization."""
    return cast(
        dict[OpOverload, Callable[..., torch.Tensor]],
        {
            **getattr(_decomp, "core_aten_decompositions")(),
            **cast(_TensorDecompositions, _decomp).get_decompositions(operators),
        },
    )


def check_functional_aten(target: object) -> None:
    if not isinstance(target, OpOverload):
        raise ValueError(f"export requires ATen operators, got {target}")
    schema: object = getattr(cast(object, target), "_schema")
    if not isinstance(schema, torch.FunctionSchema):
        raise ValueError("export operator has no supported schema")
    if target.namespace != "aten" or schema.is_mutable:
        raise ValueError(f"export contains a nonfunctional or external operator: {target}")
    if torch.Tag.nondeterministic_seeded in target.tags:
        raise ValueError("export RNG needs explicit instance-addressed inputs")


def tensor_node_metadata(node: torch.fx.Node) -> FakeTensor | None:
    value: object = node.meta.get("val")
    return value if isinstance(value, FakeTensor) else None


def load_tensor_fields(path: Path, fields: tuple[str, ...]) -> tuple[torch.Tensor, ...]:
    """Decode a weights-only checkpoint into the caller's explicit tensor field order."""
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid tensor checkpoint: {path.name}")
    values = cast(dict[object, object], payload)
    tensors: list[torch.Tensor] = []
    for field in fields:
        value = values.get(field)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"invalid tensor checkpoint field {field}: {path.name}")
        tensors.append(value)
    return tuple(tensors)


class _CudaDevice(Protocol):
    uuid: object


class _CudaAllocator(Protocol):
    def memory_snapshot(self, mempool_id: tuple[int, int], include_traces: bool) -> list[dict[str, object]]: ...
    def get_device_properties(self, device: torch.device) -> _CudaDevice: ...


def cuda_device_uuid(device: torch.device) -> str:
    return str(UUID(str(cast(_CudaAllocator, torch.cuda).get_device_properties(device).uuid)))


CudaStorageHandle = tuple[int, bytes, int, int, bytes, int, bytes, bool]


class _CudaStorage(Protocol):
    def _share_cuda_(self) -> CudaStorageHandle: ...

    @staticmethod
    def _new_shared_cuda(
        device: int,
        handle: bytes,
        size: int,
        offset: int,
        counter: bytes,
        counter_offset: int,
        event: bytes,
        wait: bool,
    ) -> torch.UntypedStorage: ...

    def share(self) -> CudaStorageHandle:
        return self._share_cuda_()

    def open(self, handle: CudaStorageHandle) -> torch.UntypedStorage:
        return self._new_shared_cuda(*handle)


def share_cuda_storage(tensor: torch.Tensor) -> CudaStorageHandle:
    return _CudaStorage.share(cast(_CudaStorage, tensor.untyped_storage()))


def open_cuda_storage(handle: CudaStorageHandle) -> torch.UntypedStorage:
    return _CudaStorage.open(cast(_CudaStorage, torch.UntypedStorage), handle)


def cuda_graph_pool_bytes(pool: tuple[int, int], device: int) -> int:
    """Decode PyTorch allocator facts at the foreign-data boundary, only in cold planning."""
    if torch.cuda.get_allocator_backend() != "native" or pool == (0, 0):
        raise ValueError("CUDA graph residency requires its captured native allocator pool")
    segments = cast(_CudaAllocator, torch.cuda).memory_snapshot(pool, False)
    total = 0
    for segment in segments:
        if _primitive(segment["device"], int) == device:
            size = _primitive(segment["total_size"], int)
            if size < 0:
                raise ValueError("CUDA graph pool size must be nonnegative")
            total += size
    return total


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
    payload = cast(dict[str, object], json.loads(path.read_text()))
    events = cast(list[dict[str, object]], payload["traceEvents"])
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
        duration = _primitive(event["dur"], float)
        start = _primitive(event["ts"], float)
        device_event = category in ("kernel", "gpu_memcpy", "gpu_memset")
        key = (str(event["pid"]), "device" if device_event else str(event["tid"]))
        interval = (start, start + duration)
        spans["gpu" if device_event else "api"].setdefault(key, []).append(interval)
        if category == "kernel":
            kernels += 1
            kernel_us += duration
        elif category == "gpu_memcpy":
            args = cast(dict[str, object], event["args"])
            transfer += _primitive(args["bytes"], int)
        elif not device_event:
            name = _primitive(event["name"], str).lower()
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
