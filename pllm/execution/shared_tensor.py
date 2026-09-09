"""Shared storage and completion primitives; no state or version ownership."""

from __future__ import annotations

from multiprocessing.connection import Connection
from contextlib import ExitStack
from dataclasses import dataclass, replace
from math import prod
import mmap
import os
import stat
from tempfile import TemporaryFile

import torch
from pllm.execution.codec import cuda_device_uuid, share_cuda_storage, open_cuda_storage
from pllm.contracts.tensor_layout import LayoutFieldId, MutationGranularity, TensorDType, TensorFieldSpec


TORCH_DTYPES = {
    TensorDType.BOOL: torch.bool, TensorDType.UINT8: torch.uint8,
    TensorDType.INT32: torch.int32, TensorDType.INT64: torch.int64,
    TensorDType.FLOAT16: torch.float16, TensorDType.BFLOAT16: torch.bfloat16,
    TensorDType.FLOAT32: torch.float32, TensorDType.FLOAT64: torch.float64,
}


def tensor_dtype(dtype: TensorDType) -> torch.dtype:
    return TORCH_DTYPES[dtype]


@dataclass(frozen=True, slots=True)
class CudaIpcStorage:
    """PyTorch's storage IPC ABI, with no tensor shape or logical state copy."""

    device: int
    device_uuid: str
    handle: str
    size_bytes: int
    offset_bytes: int
    counter_handle: str
    counter_offset: int
    event_handle: str
    event_sync_required: bool

    @classmethod
    def export(cls, tensor: torch.Tensor, stream: torch.cuda.Stream) -> CudaIpcStorage:
        if tensor.device.type != "cuda" or tensor.device != stream.device:
            raise ValueError("CUDA IPC storage requires a tensor on its original stream")
        if not tensor.numel():
            return cls(stream.device.index or 0, cuda_device_uuid(tensor.device), "", 0, 0, "", 0, "", False)
        with torch.cuda.stream(stream):
            device, handle, size, offset, counter, counter_offset, event, wait = share_cuda_storage(tensor)
        return cls(device, cuda_device_uuid(tensor.device),
            handle.hex(), size, offset, counter.hex(), counter_offset, event.hex(), wait)

    def map(self, size_bytes: int, dtype: torch.dtype, offset: int,
            shape: tuple[int, ...], stream: torch.cuda.Stream, stride: tuple[int, ...] | None = None) -> torch.Tensor:
        handle, counter, event = (bytes.fromhex(value) for value in (self.handle, self.counter_handle, self.event_handle))
        element_size = torch.empty(0, dtype=dtype).element_size()
        stride = tuple(prod(shape[index + 1:]) for index in range(len(shape))) if stride is None else stride
        if (stream.device != torch.device("cuda", self.device)
            or cuda_device_uuid(stream.device) != self.device_uuid):
            raise ValueError("CUDA IPC receiving device differs from the original GPU")
        if self.size_bytes != size_bytes:
            raise ValueError(f"CUDA IPC storage has {self.size_bytes} bytes; signed pool requires {size_bytes}")
        if (len(stride) != len(shape) or any(value < 0 for value in (*shape, *stride)) or offset < 0):
            raise ValueError("CUDA IPC tensor shape, stride or offset is invalid")
        if size_bytes == 0 and prod(shape) == 0 and not any((handle, counter, event, offset,
            self.offset_bytes, self.counter_offset, self.event_sync_required)):
            return torch.empty_strided(shape, stride, dtype=dtype, device=stream.device)
        extent = 0 if not prod(shape) else 1 + sum((dimension - 1) * step for dimension, step in zip(shape, stride, strict=True))
        # PyTorch allocator headers make this larger than a raw CUDA IPC handle.
        if (self.offset_bytes < 0 or self.counter_offset < 0 or not counter
            or not 0 < len(handle) <= 4096 or (self.event_sync_required and len(event) != 64)
            or (offset + extent) * element_size > size_bytes):
            raise ValueError("CUDA IPC handle or tensor range differs from its signed storage")
        with torch.cuda.stream(stream):
            storage = open_cuda_storage((self.device, handle, self.size_bytes, self.offset_bytes,
                counter, self.counter_offset, event, self.event_sync_required))
            return torch.empty(0, dtype=dtype, device=stream.device).set_(storage, offset, shape, stride)


@dataclass(frozen=True, slots=True)
class CpuFileStorage:
    descriptor: int
    inode: int


@dataclass(frozen=True, slots=True)
class SharedTensor:
    """Original input/output storage, retained until the existing worker acknowledgement."""

    process_id: int
    field: TensorFieldSpec
    storage: CpuFileStorage | CudaIpcStorage
    offset: int = 0
    stride: tuple[int, ...] = ()

    @classmethod
    def allocate(cls, field: TensorFieldSpec, lifetime: ExitStack,
                 device: torch.device = torch.device("cpu")) -> tuple[torch.Tensor, SharedTensor]:
        size = cls._size(field)
        if device.type == "cuda":
            tensor = torch.empty(field.shape, device=device, dtype=tensor_dtype(field.dtype))
            if tensor.data_ptr() % field.alignment_bytes:
                raise ValueError("CUDA output allocation does not meet its declared alignment")
            return tensor, replace(cls.export(tensor, lifetime), field=field)
        if device.type != "cpu":
            raise ValueError("shared output storage requires CPU or CUDA")
        backing = lifetime.enter_context(TemporaryFile())
        backing.truncate(size)
        tensor = (map_cpu_storage(backing.fileno(), size, tensor_dtype(field.dtype), 0, field.shape)
            if size else torch.empty(field.shape, dtype=tensor_dtype(field.dtype)))
        return tensor, cls(os.getpid(), field, CpuFileStorage(backing.fileno(), os.fstat(backing.fileno()).st_ino))

    @classmethod
    def export(cls, tensor: torch.Tensor, lifetime: ExitStack) -> SharedTensor:
        if tensor.device.type not in ("cpu", "cuda") or tensor.layout != torch.strided:
            raise ValueError("shared auxiliary input requires a strided CPU or CUDA tensor")
        field = cls.describe(tensor)
        if tensor.device.type == "cuda":
            source = tensor.detach()
            # The original generate ExitStack retains this view through FINISH/ABORT.
            lifetime.callback(lambda: source)
            storage = CudaIpcStorage.export(source, torch.cuda.current_stream(source.device))
            return cls(os.getpid(), field, storage, int(source.storage_offset()) if source.numel() else 0, tuple(source.stride()))
        shared, descriptor = cls.allocate(field, lifetime)
        with torch.no_grad():
            shared.copy_(tensor.detach())
        return descriptor

    @staticmethod
    def describe(tensor: torch.Tensor) -> TensorFieldSpec:
        dtype = next((dtype for dtype, native in TORCH_DTYPES.items() if native == tensor.dtype), None)
        if dtype is None or tensor.layout != torch.strided:
            raise ValueError("auxiliary tensor must use a strided tensor in the closed dtype vocabulary")
        return TensorFieldSpec(LayoutFieldId(0), tuple(tensor.shape), dtype,
            tensor.element_size(), MutationGranularity.WHOLE_TENSOR)

    @staticmethod
    def _size(field: TensorFieldSpec) -> int:
        if (any(dimension < 0 for dimension in field.shape) or field.alignment_bytes < 1
            or mmap.ALLOCATIONGRANULARITY % field.alignment_bytes):
            raise ValueError("shared tensor shape or mapping alignment is invalid")
        return prod(field.shape) * torch.empty(0, dtype=tensor_dtype(field.dtype)).element_size()

    def map(self, peer_pid: int, stream: torch.cuda.Stream | None = None) -> torch.Tensor:
        if self.process_id != peer_pid or peer_pid < 1:
            raise ValueError("shared tensor is not owned by the original IPC peer")
        size = self._size(self.field)
        storage = self.storage
        if isinstance(storage, CudaIpcStorage):
            if stream is None:
                raise ValueError("shared CUDA tensor requires its receiving stream")
            return storage.map(storage.size_bytes, tensor_dtype(self.field.dtype), self.offset,
                self.field.shape, stream, self.stride)
        if storage.descriptor < 0 or self.offset or self.stride:
            raise ValueError("CPU mapping requires its original owned contiguous storage")
        fd = os.open(f"/proc/{peer_pid}/fd/{storage.descriptor}", os.O_RDWR)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_ino != storage.inode or info.st_size != size:
                raise ValueError("shared tensor allocation identity or extent changed")
            return (map_cpu_storage(fd, size, tensor_dtype(self.field.dtype), 0, self.field.shape)
                if size else torch.empty(self.field.shape, dtype=tensor_dtype(self.field.dtype)))
        finally:
            os.close(fd)


def map_cpu_storage(fd: int, size: int, dtype: torch.dtype, offset: int, shape: tuple[int, ...]) -> torch.Tensor:
    mapping = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
    element_size = torch.empty(0, dtype=dtype).element_size()
    return torch.frombuffer(mapping, dtype=dtype, count=prod(shape), offset=offset * element_size).reshape(shape)


class RemoteCompletion:
    """One exclusive byte channel acknowledges one exact exported access."""

    def __init__(self, connection: Connection, expected: bytes) -> None:
        self.connection = connection
        self._expected = expected
        self._complete = False
        self._failure: str | None = None

    def query(self, timeout: float = 0.0) -> bool:
        if self._failure is not None:
            raise RuntimeError(self._failure)
        if not self._complete:
            try:
                if not self.connection.poll(timeout):
                    return False
                if self.connection.recv_bytes(len(self._expected)) != self._expected:
                    raise ValueError("remote access acknowledged another export")
            except (EOFError, OSError, ValueError) as exc:
                self._failure = "remote access has no valid completion proof"
                raise RuntimeError(self._failure) from exc
            self._complete = True
        return self._complete

    def synchronize(self) -> None:
        if not self.query(30):
            raise TimeoutError("remote access is still pending; its original owner remains retained")
