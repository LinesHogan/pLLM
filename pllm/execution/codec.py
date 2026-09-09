"""Closed serialization of the original contracts, never Python type paths."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import torch
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Protocol, TypeVar, cast, get_origin
from uuid import UUID

from cattrs import Converter
from cattrs.errors import ForbiddenExtraKeysError
from cattrs.strategies import configure_tagged_union

from pllm.contracts.semantic_graph import (
    ConditionalLayerReplay, TokenwiseLayerReplay,
    HiddenInputSource, HostServiceInputSource, PackedRowsInputSource, PackedSamplerRowsInputSource,
    SamplerFeedbackInputSource, SemanticInputSource, StateSelectionInputSource, StepIdInputSource, TaskResultInputSource,
)


T = TypeVar("T")
E = TypeVar("E", bound=StrEnum)
_converter = Converter(forbid_extra_keys=True, detailed_validation=False)
_converter.register_unstructure_hook(torch.device, str)


class _TensorHooks(Protocol):
    def register_hook(self, hook: Callable[[torch.Tensor], torch.Tensor]) -> torch.utils.hooks.RemovableHandle: ...


def register_gradient_hook(tensor: torch.Tensor, hook: Callable[[torch.Tensor], torch.Tensor]) -> None:
    cast(_TensorHooks, tensor).register_hook(hook)


def _primitive(value: object, target: type[T]) -> T:
    if target is float and type(value) is int:
        return cast(T, float(value))
    if type(value) is not target:
        raise TypeError("boundary scalar has the wrong type")
    return cast(T, value)


for _type in (str, int, float, bool):
    _converter.register_structure_hook(_type, _primitive)


def _enum(value: object, target: type[E]) -> E:
    return target(_primitive(value, str))


_converter.register_structure_hook(StrEnum, _enum)

# Keep cattrs' typed tuple handler, but reject iterable objects that are not JSON arrays.
_tuple_handler = _converter.get_structure_hook(tuple)


def _array(value: object, target: object) -> object:
    if type(value) is not list:
        raise TypeError("boundary sequence must be a JSON array")
    return _tuple_handler(value, target)


_converter.register_structure_hook_func(lambda target: get_origin(target) is tuple, _array)

_input_tags: dict[type[object], str] = {
    HiddenInputSource: "hidden", StepIdInputSource: "step_id", PackedRowsInputSource: "packed_rows",
    PackedSamplerRowsInputSource: "packed_sampler_rows", SamplerFeedbackInputSource: "sampler_feedback",
    HostServiceInputSource: "host_service", StateSelectionInputSource: "state_selection", TaskResultInputSource: "result",
}
configure_tagged_union(SemanticInputSource, _converter, tag_generator=_input_tags.__getitem__, tag_name="kind")


def _replay(value: object, _target: object) -> ConditionalLayerReplay | TokenwiseLayerReplay | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise TypeError("replay effect must be a closed object")
    if "layer" in value:
        return _converter.structure(value, ConditionalLayerReplay)
    return _converter.structure(value, TokenwiseLayerReplay)


_converter.register_structure_hook(ConditionalLayerReplay | TokenwiseLayerReplay | None, _replay)


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate boundary field")
    return result


def _nonfinite(value: str) -> object:
    raise ValueError("non-finite boundary scalar: " + value)


def encode_boundary(value: object) -> str:
    return json.dumps(_converter.unstructure(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def boundary_digest(*values: object) -> str:
    """Bind original typed parameters and cold facts with the same closed encoding."""
    return sha256(encode_boundary(values).encode("utf-8")).hexdigest()


def decode_boundary(payload: str, target: type[T]) -> T:
    """The caller supplies a statically bound type, never a type named by payload."""
    try:
        return _converter.structure(json.loads(payload, object_pairs_hook=_object, parse_constant=_nonfinite), target)
    except (TypeError, AttributeError, KeyError, ValueError, ForbiddenExtraKeysError) as exc:
        raise ValueError("invalid closed boundary payload") from exc


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
    def _new_shared_cuda(device: int, handle: bytes, size: int, offset: int,
                         counter: bytes, counter_offset: int, event: bytes,
                         wait: bool) -> torch.UntypedStorage: ...

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
        kind: {} for kind in ("api", "enqueue", "wait", "gpu", "launch", "graph")}
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in ("kernel", "cuda_runtime", "cuda_driver", "gpu_memcpy", "gpu_memset"):
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
    return CudaProfileTotals(kernels, kernel_us, totals["api"][1], totals["enqueue"][1],
        totals["wait"][1], totals["gpu"][1], transfer, totals["launch"][0], totals["graph"][0])
