"""Shared storage and completion primitives; no state or version ownership."""

from __future__ import annotations

import mmap
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass, replace
from math import prod
from multiprocessing.connection import Connection
from tempfile import TemporaryFile

import torch

from pllm.contracts.tensor_layout import LayoutFieldId, TensorDType, TensorFieldSpec
from pllm.execution.torch_private import cuda_device_uuid, open_cuda_storage, share_cuda_storage


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
        assert tensor.device.type == "cuda" and tensor.device == stream.device
        if not tensor.numel():
            return cls(stream.device.index or 0, cuda_device_uuid(tensor.device), "", 0, 0, "", 0, "", False)
        with torch.cuda.stream(stream):
            device, handle, size, offset, counter, counter_offset, event, wait = share_cuda_storage(tensor)
        return cls(
            device,
            cuda_device_uuid(tensor.device),
            handle.hex(),
            size,
            offset,
            counter.hex(),
            counter_offset,
            event.hex(),
            wait,
        )

    def map(
        self,
        size_bytes: int,
        dtype: torch.dtype,
        offset: int,
        shape: tuple[int, ...],
        stream: torch.cuda.Stream,
        stride: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if stream.device != torch.device("cuda", self.device) or cuda_device_uuid(stream.device) != self.device_uuid:
            raise ValueError("CUDA IPC receiving device differs from the original GPU")
        stride = tuple(prod(shape[index + 1 :]) for index in range(len(shape))) if stride is None else stride
        if not size_bytes and not prod(shape):
            return torch.empty_strided(shape, stride, dtype=dtype, device=stream.device)
        handle, counter, event = (
            bytes.fromhex(value) for value in (self.handle, self.counter_handle, self.event_handle)
        )
        with torch.cuda.stream(stream):
            storage = open_cuda_storage(
                (
                    self.device,
                    handle,
                    self.size_bytes,
                    self.offset_bytes,
                    counter,
                    self.counter_offset,
                    event,
                    self.event_sync_required,
                )
            )
            return torch.empty(0, dtype=dtype, device=stream.device).set_(storage, offset, shape, stride)

    def open_field(
        self,
        peer_pid: int,
        field: TensorFieldSpec,
        offset: int,
        stride: tuple[int, ...],
        stream: torch.cuda.Stream | None,
    ) -> torch.Tensor:
        assert stream is not None
        return self.map(self.size_bytes, field.dtype.torch, offset, field.shape, stream, stride)


@dataclass(frozen=True, slots=True)
class CpuFileStorage:
    descriptor: int
    inode: int

    def open_field(
        self,
        peer_pid: int,
        field: TensorFieldSpec,
        offset: int,
        stride: tuple[int, ...],
        stream: torch.cuda.Stream | None,
    ) -> torch.Tensor:
        assert self.descriptor >= 0 and not offset and not stride
        size = prod(field.shape) * field.dtype.itemsize
        fd = os.open(f"/proc/{peer_pid}/fd/{self.descriptor}", os.O_RDWR)
        try:
            info = os.fstat(fd)
            assert stat.S_ISREG(info.st_mode) and info.st_ino == self.inode and info.st_size == size
            return (
                map_cpu_storage(fd, size, field.dtype.torch, 0, field.shape)
                if size
                else torch.empty(field.shape, dtype=field.dtype.torch)
            )
        finally:
            os.close(fd)


@dataclass(frozen=True, slots=True)
class SharedTensor:
    """Original input/output storage, retained until the existing worker acknowledgement."""

    process_id: int
    field: TensorFieldSpec
    storage: CpuFileStorage | CudaIpcStorage
    offset: int = 0
    stride: tuple[int, ...] = ()

    @classmethod
    def allocate(
        cls, field: TensorFieldSpec, lifetime: ExitStack, device: torch.device = torch.device("cpu")
    ) -> tuple[torch.Tensor, SharedTensor]:
        assert device.type in ("cpu", "cuda")
        if device.type == "cuda":
            tensor = torch.empty(field.shape, device=device, dtype=field.dtype.torch)
            assert tensor.data_ptr() % field.alignment_bytes == 0
            return tensor, replace(cls.export(tensor, lifetime), field=field)
        size = prod(field.shape) * field.dtype.itemsize
        backing = lifetime.enter_context(TemporaryFile())
        backing.truncate(size)
        tensor = (
            map_cpu_storage(backing.fileno(), size, field.dtype.torch, 0, field.shape)
            if size
            else torch.empty(field.shape, dtype=field.dtype.torch)
        )
        return tensor, cls(os.getpid(), field, CpuFileStorage(backing.fileno(), os.fstat(backing.fileno()).st_ino))

    @classmethod
    def export(cls, tensor: torch.Tensor, lifetime: ExitStack) -> SharedTensor:
        field = cls.describe(tensor)
        if tensor.device.type == "cuda":
            source = tensor.detach()
            # The original generate ExitStack retains this view through FINISH/ABORT.
            lifetime.callback(lambda: source)
            storage = CudaIpcStorage.export(source, torch.cuda.current_stream(source.device))
            return cls(
                os.getpid(),
                field,
                storage,
                int(source.storage_offset()) if source.numel() else 0,
                tuple(source.stride()),
            )
        shared, descriptor = cls.allocate(field, lifetime)
        with torch.no_grad():
            shared.copy_(tensor.detach())
        return descriptor

    @staticmethod
    def describe(tensor: torch.Tensor) -> TensorFieldSpec:
        dtype = next((dtype for dtype in TensorDType if dtype.torch == tensor.dtype), None)
        if dtype is None or tensor.layout != torch.strided or tensor.device.type not in ("cpu", "cuda"):
            raise ValueError("host batch tensors must be strided CPU or CUDA tensors of a supported dtype")
        return TensorFieldSpec(LayoutFieldId(0), tuple(tensor.shape), dtype, tensor.element_size())

    def map(self, peer_pid: int, stream: torch.cuda.Stream | None = None) -> torch.Tensor:
        if self.process_id != peer_pid or peer_pid < 1:
            raise ValueError("shared tensor is not owned by the original IPC peer")
        return self.storage.open_field(peer_pid, self.field, self.offset, self.stride, stream)


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
        if not self._complete and self._failure is None:
            try:
                if not self.connection.poll(timeout):
                    return False
                if self.connection.recv_bytes(len(self._expected)) != self._expected:
                    self._failure = "remote access has no valid completion proof"
            except (EOFError, OSError):
                self._failure = "remote access has no valid completion proof"
            self._complete = self._failure is None
        if self._failure is not None:
            raise RuntimeError(self._failure)
        return self._complete

    def synchronize(self) -> None:
        if not self.query(30):
            raise TimeoutError("remote access is still pending; its original owner remains retained")
