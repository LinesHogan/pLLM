"""Quarantined decoding of external Torch runtime value names."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch.utils import _pytree


class CudaLongHostStaging:
    """Shape-keyed pinned storage with explicit H2D reuse completion."""

    __slots__ = (
        "_capacity",
        "_device",
        "_host",
        "_events",
        "_active_slots",
        "_in_flight",
        "_ready",
        "_capture_slots",
        "_channels",
        "_last_values",
    )

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        *,
        capture_slots: int = 8,
        channels: int = 1,
    ) -> None:
        if capacity <= 0:
            raise ValueError("CUDA host staging capacity must be positive")
        if capture_slots <= 0:
            raise ValueError("CUDA host staging requires capture slots")
        if channels <= 0:
            raise ValueError("CUDA host staging requires channels")
        self._capacity = capacity
        self._device = device
        self._capture_slots = capture_slots
        self._channels = channels
        cuda = device.type == "cuda"
        self._host = torch.empty(
            (capacity + 1, capture_slots, channels, capacity),
            dtype=torch.long,
            pin_memory=cuda,
        )
        self._events = tuple(
            torch.cuda.Event(external=True) for _slot in range(capacity + 1)
        ) if cuda else ()
        self._active_slots = [0] * (capacity + 1)
        self._in_flight = [False] * (capacity + 1)
        self._ready = [False] * (capacity + 1)
        self._last_values: list[
            tuple[tuple[int, ...], ...] | None
        ] = [None] * (capacity + 1)

    def copy(self, values: list[int], target: torch.Tensor) -> None:
        self.copy_channels(((values, target),))

    def copy_channels(
        self,
        channels: tuple[tuple[list[int], torch.Tensor], ...],
    ) -> None:
        if len(channels) != self._channels:
            raise RuntimeError("CUDA host staging channel count differs")
        counts = {len(values) for values, _target in channels}
        if len(counts) != 1:
            raise RuntimeError("CUDA host staging channels must align")
        count = next(iter(counts))
        if count > self._capacity or any(
            target.numel() < count for _values, target in channels
        ):
            raise RuntimeError("CUDA host staging value count exceeds capacity")
        cuda = self._device.type == "cuda"
        capturing = cuda and torch.cuda.is_current_stream_capturing()
        content = tuple(tuple(values) for values, _target in channels)
        if (
            not capturing
            and self._active_slots[count] > 0
            and self._last_values[count] == content
        ):
            return
        if capturing:
            slot = self._active_slots[count]
            if slot >= self._capture_slots:
                raise RuntimeError(
                    "CUDA host staging exhausted capture slots for one shape"
                )
            self._active_slots[count] += 1
            self._ready[count] = False
            hosts = (self._host[count, slot],)
        else:
            if cuda and self._in_flight[count]:
                if not self._ready[count]:
                    raise RuntimeError(
                        "CUDA host staging was reused before sampler completion"
                    )
                event = self._events[count]
                if not event.query():
                    event.synchronize()
                self._ready[count] = False
                self._in_flight[count] = False
            if self._active_slots[count] == 0:
                self._active_slots[count] = 1
            hosts = tuple(
                self._host[count, slot]
                for slot in range(self._active_slots[count])
            )
        for host in hosts:
            for channel, (values, _target) in enumerate(channels):
                for position, value in enumerate(values):
                    host[channel, position] = value
        for channel, (_values, target) in enumerate(channels):
            target[:count].copy_(
                hosts[0][channel, :count],
                non_blocking=cuda,
            )
        self._in_flight[count] = True
        self._last_values[count] = content
        if cuda:
            self._events[count].record(torch.cuda.current_stream(self._device))
            self._ready[count] = True


def torch_assert_async(condition: torch.Tensor, message: str) -> None:
    """Invoke Torch's compiler-aware device assertion in one quarantine."""

    dynamic = cast(Any, torch)
    dynamic._assert_async(condition, message)


def torch_record_stream_owners(
    values: tuple[object, ...], stream: object
) -> None:
    """Transfer opaque Tensor storage lifetime to one CUDA stream."""

    if not isinstance(stream, torch.cuda.Stream):
        return
    leaves = cast(list[object], _pytree.tree_leaves(values))
    for value in leaves:
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            value.record_stream(stream)


def initialize_adam_capture_storage(
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
) -> None:
    """Create address-stable Adam state and gradients before CUDA capture."""

    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
        state = optimizer.state[parameter]
        state["step"] = torch.zeros(
            (), dtype=torch.float32, device=parameter.device
        )
        state["exp_avg"] = torch.zeros_like(parameter)
        state["exp_avg_sq"] = torch.zeros_like(parameter)


def torch_cuda_tensor_tuple(
    values: tuple[object, ...],
) -> tuple[torch.Tensor, ...]:
    """Narrow a signed Tensor port tuple at the Torch runtime boundary."""

    if any(
        not isinstance(value, torch.Tensor) or value.device.type != "cuda"
        for value in values
    ):
        raise RuntimeError("captured Tensor batch requires CUDA Tensor inputs")
    return cast(tuple[torch.Tensor, ...], values)


def torch_cuda_stream(value: object) -> torch.cuda.Stream:
    if not isinstance(value, torch.cuda.Stream):
        raise RuntimeError("captured Tensor batch requires a CUDA stream")
    return value


def torch_is_current_stream_capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def torch_regular_position_view(
    tensor: torch.Tensor, positions: list[int]
) -> torch.Tensor | None:
    """Return an allocation-free strided view for a regular position list."""

    if not positions:
        return tensor[:0]
    if tensor.ndim != 1:
        return None
    if len(positions) == 1:
        position = positions[0]
        if position < 0 or position >= tensor.numel():
            return None
        return tensor[position : position + 1]
    step = positions[1] - positions[0]
    if step <= 0:
        return None
    start = positions[0]
    last = positions[-1]
    if start < 0 or last >= tensor.numel():
        return None
    if any(
        right - left != step
        for left, right in zip(positions, positions[1:], strict=False)
    ):
        return None
    return tensor.as_strided(
        (len(positions),),
        (tensor.stride(0) * step,),
        storage_offset=tensor.storage_offset() + start * tensor.stride(0),
    )


def runtime_torch_values(
    device_wire: str, dtype_wire: str
) -> tuple[torch.device, torch.dtype]:
    if not dtype_wire.startswith("torch."):
        raise RuntimeError("worker returned an invalid Torch dtype specialization")
    dtype = cast(Any, getattr(torch, dtype_wire.removeprefix("torch."), None))
    if not isinstance(dtype, torch.dtype) or str(dtype) != dtype_wire:
        raise RuntimeError("worker returned an invalid Torch dtype specialization")
    try:
        device = torch.device(device_wire)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("worker returned an invalid Torch device specialization") from exc
    if str(device) != device_wire:
        raise RuntimeError("worker returned a noncanonical Torch device specialization")
    return device, dtype


__all__ = [
    "runtime_torch_values",
    "CudaLongHostStaging",
    "initialize_adam_capture_storage",
    "torch_record_stream_owners",
    "torch_cuda_stream",
    "torch_cuda_tensor_tuple",
    "torch_is_current_stream_capturing",
    "torch_regular_position_view",
    "torch_assert_async",
]
