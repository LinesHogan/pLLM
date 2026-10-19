"""Small CUDA helpers: stream readiness between producer and consumer, and non-blocking uploads."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def upload(
    values: Sequence[int] | Sequence[Sequence[int]], device: torch.device, dtype: torch.dtype = torch.long
) -> torch.Tensor:
    """Copy small host index lists to `device` without blocking the host.

    `torch.tensor(values, device=cuda)` copies from pageable memory, which makes the host wait for
    the stream. Staging through pinned memory lets the copy queue behind the GPU work instead.
    """
    host = torch.tensor(values, dtype=dtype)
    return host.pin_memory().to(device, non_blocking=True) if device.type == "cuda" else host


class CudaStreamReadiness:
    __slots__ = ("_producer_stream", "_event")

    def __init__(self, producer_stream: torch.cuda.Stream) -> None:
        if type(producer_stream) is not torch.cuda.Stream:
            raise TypeError("CUDA readiness requires an owning producer stream")
        event = torch.cuda.Event(blocking=False)
        event.record(producer_stream)
        self._producer_stream = producer_stream
        self._event = event

    @property
    def device(self) -> torch.device:
        return self._producer_stream.device

    def order_on(self, consumer_stream: torch.cuda.Stream) -> None:
        if type(consumer_stream) is not torch.cuda.Stream or consumer_stream.device != self._producer_stream.device:
            raise RuntimeError("CUDA readiness consumer uses another device")
        consumer_stream.wait_event(self._event)

    def synchronize(self) -> None:
        self._event.synchronize()


__all__ = ["CudaStreamReadiness", "upload"]
