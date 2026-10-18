"""Opaque producer-to-consumer CUDA stream readiness capability."""

from __future__ import annotations

import torch


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


__all__ = ["CudaStreamReadiness"]
