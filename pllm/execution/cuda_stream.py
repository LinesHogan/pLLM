"""Small CUDA helpers: stream readiness between producer and consumer, non-blocking uploads and device counters.

The PTTS arena and the tensor executor upload small index lists with `upload`; step capture reuses
`PinnedStaging` for its per-step index table; state execution keeps diagnostics in `DeviceCounters`.
None of them makes the host wait for the stream.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import torch

from pllm.execution.admission import WindowBackpressure


def upload(
    values: Sequence[int] | Sequence[Sequence[int]] | npt.NDArray[np.int64],
    device: torch.device,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Copy small host index lists to `device` without blocking the host.

    `torch.tensor(values, device=cuda)` copies from pageable memory, which makes the host wait for
    the stream. Staging through pinned memory lets the copy queue behind the GPU work instead.
    """
    host = torch.tensor(values, dtype=dtype)
    return host.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else host.to(device)


class PinnedStaging:
    """Pinned host buffers reused for one recurring upload, such as a per-step index table.

    `torch.Tensor.pin_memory()` allocates through the host caching allocator on every call. Here a
    buffer is refilled once the copy that last read it has finished; that is a non-blocking event query,
    and while every buffer is still in flight a new one is added instead of waiting.
    """

    def __init__(self, *, capacity: int = 0, slots: int = 0) -> None:
        self._bounded = slots > 0
        self._buffers: list[tuple[torch.Tensor, torch.cuda.Event]] = []
        self._buffers.extend(
            (torch.empty(capacity, dtype=torch.long, pin_memory=True), torch.cuda.Event()) for _ in range(slots)
        )
        self._current: tuple[torch.Tensor, torch.cuda.Event] | None = None

    def take(self, size: int) -> torch.Tensor:
        """A zeroed int64 pinned host buffer of `size` elements, valid until the next `take`."""
        self._current = next((item for item in self._buffers if item[0].numel() >= size and item[1].query()), None)
        if self._current is None:
            if self._bounded:
                raise WindowBackpressure("pinned staging is still in flight; retry admission after completion")
            # Idle buffers that are too small are dropped; capacities grow in powers of two.
            self._buffers = [item for item in self._buffers if item[0].numel() >= size or not item[1].query()]
            capacity = 1 << max(size - 1, 0).bit_length()
            self._current = (torch.empty(capacity, dtype=torch.long, pin_memory=True), torch.cuda.Event())
            self._buffers.append(self._current)
        host = self._current[0][:size]
        host.zero_()
        return host

    def copy_to(self, destination: torch.Tensor) -> None:
        """Queue the copy of the taken buffer into `destination` on its device's current stream."""
        assert self._current is not None
        host, copied = self._current
        destination.copy_(host[: destination.numel()].view(destination.shape), non_blocking=True)
        copied.record(torch.cuda.current_stream(destination.device))


class CudaStreamReadiness:
    """A point on a producer stream that consumers on other streams wait for without blocking the host.

    Created right after the producer queues its work; `order_on` makes a consumer stream wait for it.
    """

    __slots__ = ("_producer_stream", "_event")

    def __init__(self, producer_stream: torch.cuda.Stream) -> None:
        event = torch.cuda.Event(blocking=False)
        event.record(producer_stream)
        self._producer_stream = producer_stream
        self._event = event

    @property
    def device(self) -> torch.device:
        return self._producer_stream.device

    def order_on(self, consumer_stream: torch.cuda.Stream) -> None:
        assert consumer_stream.device == self._producer_stream.device
        consumer_stream.wait_event(self._event)

    def synchronize(self) -> None:
        self._event.synchronize()


class DeviceCounters:
    """Two cumulative diagnostics with one ordered asynchronous host snapshot."""

    def __init__(self, device: torch.device) -> None:
        self.device = torch.zeros(2, dtype=torch.int64, device=device)
        self.host = torch.zeros(2, dtype=torch.int64, pin_memory=device.type == "cuda")
        self.ready: torch.cuda.Event | None = None
        self.values = (0, 0)

    def add(self, values: torch.Tensor, stream: torch.cuda.Stream | None) -> None:
        """Order GPU writes to the snapshot; CPU reads only after the latest event."""
        if stream is None and self.device.is_cuda:
            stream = torch.cuda.current_stream(self.device.device)
        if stream is None:
            self.device.add_(values)
            self.host.copy_(self.device)
        else:
            with torch.cuda.stream(stream):
                self.device.record_stream(stream)
                if self.ready is not None:
                    stream.wait_event(self.ready)
                self.device.add_(values)
                self.host.copy_(self.device, non_blocking=True)
                if self.ready is None:
                    self.ready = torch.cuda.Event()
                self.ready.record(stream)

    def read(self) -> tuple[int, int]:
        """Return the last complete snapshot without waiting for GPU work."""
        if self.ready is None or self.ready.query():
            self.values = (int(self.host[0]), int(self.host[1]))
        return self.values


__all__ = ["CudaStreamReadiness", "DeviceCounters", "PinnedStaging", "upload"]
