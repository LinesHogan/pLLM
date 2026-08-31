"""Common physical completion ownership for request-slot adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.contracts.request_state import (
    RequestStateAsyncPin,
    StateEffectReceipt,
)
from tllm.runtime.request_slot_arena import RequestSlotPinPlan


@dataclass(frozen=True, slots=True)
class NoCompletionFence:
    pass


@dataclass(frozen=True, slots=True)
class CudaCompletionFence:
    event: torch.cuda.Event


CompletionFence = NoCompletionFence | CudaCompletionFence


class RequestSlotAdapter:
    """Own the physical completion fences shared by slot representations."""

    __slots__ = (
        "_capacity",
        "_device",
        "_fences",
        "_poisoned",
    )

    def __init__(self, *, capacity: int, device: torch.device) -> None:
        if capacity < 1:
            raise ValueError("request-slot adapter capacity must be positive")
        self._capacity = capacity
        self._device = device
        self._fences: dict[int, CompletionFence] = {}
        self._poisoned = False

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def device(self) -> torch.device:
        return self._device

    def poison(self) -> None:
        self._poisoned = True

    def retire(self, receipt: StateEffectReceipt) -> None:
        """Release representation state after logical RETIRE authorization."""

    def replace_epoch(self, epoch: int) -> None:
        """Advance representation epoch after the logical arena is empty."""

    def record_synchronous_fence(self) -> NoCompletionFence:
        self._require_live()
        if self._device.type == "cuda":
            raise RuntimeError(
                "CUDA request state pins require a recorded completion event"
            )
        return NoCompletionFence()

    def record_cuda_fence(
        self,
        stream: torch.cuda.Stream,
    ) -> CudaCompletionFence:
        self._require_live()
        if self._device.type != "cuda":
            raise RuntimeError("CPU request state cannot record a CUDA pin")
        if stream.device != self._device:
            raise RuntimeError("request-state completion stream uses another device")
        event = torch.cuda.Event()
        event.record(stream)
        return CudaCompletionFence(event)

    def commit_fence(
        self,
        plan: RequestSlotPinPlan,
        fence: CompletionFence,
    ) -> None:
        self._require_live()
        if plan.pin.pin_id in self._fences:
            raise RuntimeError("request-state completion fence already exists")
        self._fences[plan.pin.pin_id] = fence

    def discard_fence(self, pin: RequestStateAsyncPin) -> None:
        self._require_live()
        try:
            self._fences.pop(pin.pin_id)
        except KeyError as exc:
            raise RuntimeError("request-state completion fence is unknown") from exc

    def fence_complete(self, pin: RequestStateAsyncPin) -> bool:
        self._require_live()
        match self._fence(pin):
            case NoCompletionFence():
                return True
            case CudaCompletionFence(event=event):
                return event.query()

    def order_fence(
        self,
        pin: RequestStateAsyncPin,
        stream: torch.cuda.Stream,
    ) -> None:
        self._require_live()
        if stream.device != self._device:
            raise RuntimeError("request-state ordering stream uses another device")
        match self._fence(pin):
            case NoCompletionFence():
                raise RuntimeError("CUDA request state pin has no completion event")
            case CudaCompletionFence(event=event):
                stream.wait_event(event)

    def wait_fence(self, pin: RequestStateAsyncPin) -> None:
        self._require_live()
        match self._fence(pin):
            case NoCompletionFence():
                return
            case CudaCompletionFence(event=event):
                event.synchronize()

    def _fence(self, pin: RequestStateAsyncPin) -> CompletionFence:
        try:
            return self._fences[pin.pin_id]
        except KeyError as exc:
            raise RuntimeError("request-state completion fence is unknown") from exc

    def _require_live(self) -> None:
        if self._poisoned:
            raise RuntimeError("request-slot adapter is poisoned")


__all__ = [
    "CompletionFence",
    "CudaCompletionFence",
    "NoCompletionFence",
    "RequestSlotAdapter",
]
