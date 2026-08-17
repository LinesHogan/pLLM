#!/usr/bin/env python3
"""Shared CUDA stream pool and pipeline lane for off-critical-path side work.

Extracted 1:1 from the ESamp train engine (workstream B2) so the
choreography — serial/single/dual stream topology, rotating slot buffers,
event rings, and delayed-backward draining — becomes shared tLLM runtime
infrastructure instead of consumer-private machinery. Semantics are
byte-identical to the original implementation; error messages keep their
original wording through the ``label`` parameter.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from tllm.runtime.config import AdaptationStreamMode, normalize_adaptation_stream_mode

DEFAULT_STREAM_PRIORITY = 0

#: Queued delayed-backward work item: (pipeline slot, active rows, prompt ids).
QueueEntry = tuple[int, int, tuple[int, ...]]


def ensure_event_ring(events: list[torch.cuda.Event], size: int) -> list[torch.cuda.Event]:
    """Return ``events`` unchanged when the ring size matches, else a fresh ring."""
    return events if len(events) == size else [torch.cuda.Event(blocking=False) for _ in range(size)]


@contextmanager
def train_grad_context() -> Iterator[None]:
    """Re-enable grad bookkeeping for side training under inference-mode serving."""
    with torch.inference_mode(False), torch.enable_grad():
        yield


def copy_rows_into_buffer(dst: torch.Tensor, src: torch.Tensor, *, label: str) -> int:
    if dst.ndim != 2 or src.ndim != 2 or int(dst.shape[1]) != int(src.shape[1]):
        raise ValueError(f"{label} expects rank-2 hidden batches with matching width")
    copied = min(int(dst.shape[0]), int(src.shape[0]))
    if copied > 0:
        dst[:copied].copy_(src[:copied])
    return copied


class StreamPool:
    """Forward/train side streams with serial/single/dual topology.

    ``serial`` aliases both roles to the caller's current stream (no side
    streams at all); ``single`` shares one side stream; ``dual`` keeps two
    distinct streams at the configured priority.
    """

    __slots__ = ("mode", "priority", "forward_stream", "train_stream")

    def __init__(self, *, mode: str = "dual", priority: int = DEFAULT_STREAM_PRIORITY) -> None:
        self.mode: AdaptationStreamMode = normalize_adaptation_stream_mode(mode)
        self.priority = int(priority)
        self.forward_stream: torch.cuda.Stream | None = None
        self.train_stream: torch.cuda.Stream | None = None

    @property
    def serial(self) -> bool:
        return normalize_adaptation_stream_mode(self.mode) == "serial"

    def reset(self) -> None:
        self.forward_stream = None
        self.train_stream = None

    def ensure(self, device: torch.device) -> None:
        mode = normalize_adaptation_stream_mode(self.mode)
        if mode == "serial":
            self.forward_stream = None
            self.train_stream = None
            return
        priority = int(self.priority)
        if mode == "single":
            stream = self.forward_stream or self.train_stream or torch.cuda.Stream(device=device, priority=priority)
            self.forward_stream = stream
            self.train_stream = stream
            return
        if self.forward_stream is None or self.forward_stream is self.train_stream:
            self.forward_stream = torch.cuda.Stream(device=device, priority=priority)
        if self.train_stream is None or self.train_stream is self.forward_stream:
            self.train_stream = torch.cuda.Stream(device=device, priority=priority)

    def synchronize(self) -> None:
        if self.forward_stream is not None:
            self.forward_stream.synchronize()
        if self.train_stream is not None:
            self.train_stream.synchronize()


@dataclass(slots=True)
class PipelineBuffers:
    src: torch.Tensor
    tgt: torch.Tensor


class PipelineLane:
    """Rotating slot buffers + event rings feeding a delayed training stream.

    The lane owns the launch/drain choreography; the actual per-slot training
    step is injected as the ``train_slot`` callback to
    :meth:`enqueue_ready_and_drain`.
    """

    __slots__ = (
        "streams",
        "slots",
        "label",
        "buffers",
        "next_slot",
        "current_step_slot",
        "ready_step_slot",
        "pending_train_queue",
        "src_ready_events",
        "tgt_ready_events",
        "src_staged_events",
        "tgt_done_events",
        "slot_train_done_events",
    )

    def __init__(self, *, streams: StreamPool, slots: int, label: str) -> None:
        self.streams = streams
        self.slots = max(1, int(slots))
        self.label = str(label)
        self.buffers: PipelineBuffers | None = None
        self.next_slot = 0
        self.current_step_slot: int | None = None
        self.ready_step_slot: int | None = None
        self.pending_train_queue: deque[QueueEntry] = deque()
        self.src_ready_events: list[torch.cuda.Event] = []
        self.tgt_ready_events: list[torch.cuda.Event] = []
        self.src_staged_events: list[torch.cuda.Event] = []
        self.tgt_done_events: list[torch.cuda.Event] = []
        self.slot_train_done_events: list[torch.cuda.Event] = []

    def reset_runtime_state(self) -> None:
        self.src_ready_events = []
        self.tgt_ready_events = []
        self.src_staged_events = []
        self.tgt_done_events = []
        self.slot_train_done_events = []
        self.pending_train_queue.clear()
        self.current_step_slot = None
        self.ready_step_slot = None
        self.next_slot = 0
        self.buffers = None

    def clear_in_flight(self) -> None:
        self.pending_train_queue.clear()
        self.current_step_slot = None
        self.ready_step_slot = None

    def ensure_buffers(
        self,
        *,
        rows: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
        on_reallocated: Callable[[], None],
    ) -> bool:
        """Ensure slot buffers match the requested layout; True when reallocated."""
        slots = int(self.slots)
        want_shape = (slots, int(rows), int(hidden_size))
        buffers = self.buffers
        if (
            buffers is None
            or tuple(buffers.src.shape) != want_shape
            or buffers.src.device != device
            or buffers.src.dtype != dtype
        ):
            self.reset_runtime_state()
            self.streams.ensure(device)
            self.src_ready_events = ensure_event_ring([], slots)
            self.tgt_ready_events = ensure_event_ring([], slots)
            self.src_staged_events = ensure_event_ring([], slots)
            self.tgt_done_events = ensure_event_ring([], slots)
            self.slot_train_done_events = ensure_event_ring([], slots)
            with torch.inference_mode(False):
                self.buffers = PipelineBuffers(
                    src=torch.empty(want_shape, device=device, dtype=dtype),
                    tgt=torch.empty(want_shape, device=device, dtype=dtype),
                )
            on_reallocated()
            return True
        self.streams.ensure(device)
        self.src_ready_events = ensure_event_ring(self.src_ready_events, slots)
        self.tgt_ready_events = ensure_event_ring(self.tgt_ready_events, slots)
        self.src_staged_events = ensure_event_ring(self.src_staged_events, slots)
        self.tgt_done_events = ensure_event_ring(self.tgt_done_events, slots)
        self.slot_train_done_events = ensure_event_ring(self.slot_train_done_events, slots)
        return False

    def require_buffers(self) -> PipelineBuffers:
        if (buffers := self.buffers) is None:
            raise RuntimeError(f"{self.label} pipeline buffers are unavailable before ensure_resources initializes them")
        return buffers

    def _require_device(self, hidden: torch.Tensor, *, expected: torch.device) -> None:
        if hidden.device != expected:
            raise RuntimeError(
                f"{self.label} launch device mismatch: hidden rows are on {hidden.device}, but runtime resources are on {expected}"
            )

    def _require_forward_stream(self) -> torch.cuda.Stream | None:
        forward_stream = self.streams.forward_stream
        if forward_stream is None and not self.streams.serial:
            raise RuntimeError(f"{self.label} forward stream is unavailable before ensure_resources initializes it")
        return forward_stream

    def _launch_forward_kernel(
        self,
        *,
        buffers: PipelineBuffers,
        forward_stream: torch.cuda.Stream | None,
        slot: int,
        source_hidden: torch.Tensor,
    ) -> None:
        src_ready = self.src_ready_events[slot]
        src_staged = self.src_staged_events[slot]
        done_event = self.slot_train_done_events[slot]
        current_stream = torch.cuda.current_stream(device=source_hidden.device)
        src_ready.record(current_stream)
        if forward_stream is None:
            current_stream.wait_event(done_event)
            copy_rows_into_buffer(buffers.src[slot], source_hidden, label=f"{self.label} engine")
            src_staged.record(current_stream)
            return
        with torch.cuda.stream(forward_stream):
            forward_stream.wait_event(done_event)
            forward_stream.wait_event(src_ready)
            copy_rows_into_buffer(buffers.src[slot], source_hidden, label=f"{self.label} engine")
            src_staged.record(forward_stream)

    def launch_forward(self, source_hidden: torch.Tensor) -> None:
        if self.current_step_slot is not None or self.ready_step_slot is not None or self.pending_train_queue:
            raise RuntimeError(f"{self.label} launch order violation: previous step has not been flushed")
        buffers = self.require_buffers()
        self._require_device(source_hidden, expected=buffers.src.device)
        forward_stream = self._require_forward_stream()
        slot = int(self.next_slot % int(self.slots))
        self.next_slot = (slot + 1) % int(self.slots)
        self.current_step_slot = slot
        self.ready_step_slot = None
        self._launch_forward_kernel(
            buffers=buffers,
            forward_stream=forward_stream,
            slot=slot,
            source_hidden=source_hidden,
        )

    def _launch_target_kernel(
        self,
        *,
        buffers: PipelineBuffers,
        forward_stream: torch.cuda.Stream | None,
        slot: int,
        target_hidden: torch.Tensor,
    ) -> None:
        tgt_ready = self.tgt_ready_events[slot]
        tgt_done = self.tgt_done_events[slot]
        current_stream = torch.cuda.current_stream(device=target_hidden.device)
        tgt_ready.record(current_stream)
        if forward_stream is None:
            copy_rows_into_buffer(buffers.tgt[slot], target_hidden, label=f"{self.label} engine")
            tgt_done.record(current_stream)
            self.current_step_slot = None
            self.ready_step_slot = slot
            return
        with torch.cuda.stream(forward_stream):
            forward_stream.wait_event(tgt_ready)
            copy_rows_into_buffer(buffers.tgt[slot], target_hidden, label=f"{self.label} engine")
            tgt_done.record(forward_stream)
        self.current_step_slot = None
        self.ready_step_slot = slot

    def launch_target(self, target_hidden: torch.Tensor) -> None:
        buffers = self.require_buffers()
        self._require_device(target_hidden, expected=buffers.tgt.device)
        forward_stream = self._require_forward_stream()
        slot = self.current_step_slot
        if slot is None:
            raise RuntimeError(f"{self.label} target launch requires an active pipeline slot")
        self._launch_target_kernel(
            buffers=buffers,
            forward_stream=forward_stream,
            slot=int(slot),
            target_hidden=target_hidden,
        )

    def _launch_step_kernel(
        self,
        *,
        buffers: PipelineBuffers,
        slot: int,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> None:
        if int(source_hidden.shape[0]) != int(target_hidden.shape[0]):
            raise RuntimeError(f"{self.label} step launch requires source and target batches with the same row count")
        current_stream = torch.cuda.current_stream(device=source_hidden.device)
        current_stream.wait_event(self.slot_train_done_events[slot])
        copy_rows_into_buffer(buffers.src[slot], source_hidden, label=f"{self.label} engine")
        copy_rows_into_buffer(buffers.tgt[slot], target_hidden, label=f"{self.label} engine")
        self.src_staged_events[slot].record(current_stream)
        self.tgt_done_events[slot].record(current_stream)

    def launch_step(self, source_hidden: torch.Tensor, target_hidden: torch.Tensor) -> None:
        if self.current_step_slot is not None or self.ready_step_slot is not None or self.pending_train_queue:
            raise RuntimeError(f"{self.label} launch order violation: previous step has not been flushed")
        buffers = self.require_buffers()
        self._require_device(source_hidden, expected=buffers.src.device)
        self._require_device(target_hidden, expected=buffers.tgt.device)
        slot = int(self.next_slot % int(self.slots))
        self.next_slot = (slot + 1) % int(self.slots)
        self._launch_step_kernel(
            buffers=buffers,
            slot=slot,
            source_hidden=source_hidden,
            target_hidden=target_hidden,
        )
        self.current_step_slot = None
        self.ready_step_slot = slot

    def enqueue_ready_and_drain(
        self,
        active_rows: int,
        prompt_idxs: Sequence[int] | None,
        train_slot: Callable[[int, int, tuple[int, ...]], None],
    ) -> None:
        """Queue the ready slot (if any) and drain all pending training work."""
        buffers = self.require_buffers()
        serial = self.streams.serial
        train_stream = self.streams.train_stream
        if train_stream is None and not serial:
            raise RuntimeError(f"{self.label} train stream is unavailable before ensure_resources initializes it")
        if (slot := self.ready_step_slot) is not None:
            self.pending_train_queue.append((int(slot), int(active_rows), tuple(int(x) for x in (prompt_idxs or ()))))
            self.ready_step_slot = None
        if serial:
            current_stream = torch.cuda.current_stream(device=buffers.src.device)
            while self.pending_train_queue:
                queued_slot, queued_rows, queued_prompts = self.pending_train_queue.popleft()
                current_stream.wait_event(self.src_staged_events[queued_slot])
                current_stream.wait_event(self.tgt_done_events[queued_slot])
                with train_grad_context():
                    train_slot(queued_slot, queued_rows, queued_prompts)
                self.slot_train_done_events[queued_slot].record(current_stream)
            return
        assert train_stream is not None
        with torch.cuda.stream(train_stream):
            while self.pending_train_queue:
                queued_slot, queued_rows, queued_prompts = self.pending_train_queue.popleft()
                train_stream.wait_event(self.src_staged_events[queued_slot])
                train_stream.wait_event(self.tgt_done_events[queued_slot])
                with train_grad_context():
                    train_slot(queued_slot, queued_rows, queued_prompts)
                self.slot_train_done_events[queued_slot].record(train_stream)
