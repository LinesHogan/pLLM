#!/usr/bin/env python3
"""Shared CUDA-graph capture lane for address-stable side training.

Extracted 1:1 from the ESamp train engine (workstream B2). A
:class:`GraphCaptureLane` owns one capture/replay state machine: persistent
staging buffers, the uncaptured -> captured / disabled transitions, and the
skip/replay counters. The training loss body is injected as a closure, so the
lane is generic scheduling infrastructure rather than consumer-private
machinery; semantics and counters are byte-identical to the original ESamp
implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch

from tllm.runtime.scheduling.streams import train_grad_context

CaptureState = Literal["uncaptured", "captured", "disabled"]


class ReplayGraphMissing:
    pass


MISSING_REPLAY_GRAPH = ReplayGraphMissing()
ReplayGraphRef = torch.cuda.CUDAGraph | ReplayGraphMissing


@dataclass(slots=True)
class GraphCopyPair:
    """Persistent src/tgt staging pair feeding a captured training graph."""

    src: torch.Tensor
    tgt: torch.Tensor


@dataclass(slots=True)
class CaptureSlotBuffers:
    slot_ids: torch.Tensor
    slot_sum: torch.Tensor
    slot_cnt: torch.Tensor


@dataclass(slots=True)
class CaptureBuffers:
    pair: GraphCopyPair
    valid: torch.Tensor
    loss: torch.Tensor
    slots: CaptureSlotBuffers | None = None


@dataclass(slots=True)
class GraphCaptureLane:
    """One capture/replay state machine with staging buffers and counters."""

    rows: int = 0
    slots: int = 0
    active_rows: int = 0
    external_inputs: bool = False
    capture_state: CaptureState = "uncaptured"
    replay_graph: ReplayGraphRef = field(default_factory=lambda: MISSING_REPLAY_GRAPH)
    disable_reason: str = ""
    capture_attempt_count: int = 0
    skip_not_enabled_count: int = 0
    skip_missing_optimizer_state_count: int = 0
    skip_wrong_device_count: int = 0
    replay_attempt_count: int = 0
    replay_hit_count: int = 0
    replay_stage_miss_count: int = 0
    kernel_fallback_count: int = 0
    buffers: CaptureBuffers | None = None

    def reset(self) -> None:
        self.capture_state = "uncaptured"
        self.replay_graph = MISSING_REPLAY_GRAPH
        self.rows = 0
        self.slots = 0
        self.active_rows = 0
        self.external_inputs = False
        self.disable_reason = ""
        self.capture_attempt_count = 0
        self.skip_not_enabled_count = 0
        self.skip_missing_optimizer_state_count = 0
        self.skip_wrong_device_count = 0
        self.replay_attempt_count = 0
        self.replay_hit_count = 0
        self.replay_stage_miss_count = 0
        self.kernel_fallback_count = 0
        self.buffers = None

    def ensure_storage(
        self,
        *,
        rows: int,
        hidden: int,
        device: torch.device,
        dtype: torch.dtype,
        slots: int = 0,
        with_slot_buffers: bool = False,
        src_input: torch.Tensor | None = None,
        tgt_input: torch.Tensor | None = None,
    ) -> None:
        if (src_input is None) != (tgt_input is None):
            raise RuntimeError("external graph inputs require both source and target tensors")
        external_inputs = src_input is not None
        if self.rows == rows and self.slots == slots and self.external_inputs == external_inputs and self.buffers:
            return
        self.rows = rows
        self.slots = slots
        self.active_rows = 0
        self.external_inputs = external_inputs
        self.capture_state = "uncaptured"
        self.replay_graph = MISSING_REPLAY_GRAPH
        self.disable_reason = ""
        self.capture_attempt_count = 0
        self.skip_not_enabled_count = 0
        self.skip_missing_optimizer_state_count = 0
        self.skip_wrong_device_count = 0
        self.replay_attempt_count = 0
        self.replay_hit_count = 0
        self.replay_stage_miss_count = 0
        self.kernel_fallback_count = 0
        with torch.inference_mode(False):
            if src_input is None:
                src_buffer = torch.zeros((rows, hidden), device=device, dtype=dtype)
                tgt_buffer = torch.zeros((rows, hidden), device=device, dtype=dtype)
            else:
                assert tgt_input is not None
                src_buffer = src_input[:rows]
                tgt_buffer = tgt_input[:rows]
            slot_buffers: CaptureSlotBuffers | None = None
            if with_slot_buffers:
                slot_buffers = CaptureSlotBuffers(
                    slot_ids=torch.zeros((rows,), device=device, dtype=torch.long),
                    slot_sum=torch.zeros((slots,), device=device, dtype=torch.float32),
                    slot_cnt=torch.zeros((slots,), device=device, dtype=torch.float32),
                )
            self.buffers = CaptureBuffers(
                pair=GraphCopyPair(src=src_buffer, tgt=tgt_buffer),
                valid=torch.zeros((rows,), device=device, dtype=torch.float32),
                loss=torch.zeros((1,), device=device, dtype=torch.float32),
                slots=slot_buffers,
            )

    def capture_body(self, *, stream: torch.cuda.Stream, body: Callable[[CaptureBuffers], None]) -> bool:
        """Capture ``body`` into a CUDA graph; disable the lane on failure."""
        try:
            graph_obj = torch.cuda.CUDAGraph()
            buffers = self.buffers
            assert buffers is not None  # ensure_storage just populated them
            with torch.inference_mode():
                with torch.cuda.graph(graph_obj, stream=stream):
                    with train_grad_context():
                        body(buffers)
            self.capture_state = "captured"
            self.replay_graph = graph_obj
            self.disable_reason = ""
            return True
        except Exception as exc:
            self.capture_state = "disabled"
            self.replay_graph = MISSING_REPLAY_GRAPH
            self.disable_reason = str(exc)
            return False

    def stage_inputs(
        self,
        *,
        src: torch.Tensor,
        tgt: torch.Tensor,
        active_rows: int,
        slot_ids: torch.Tensor | None = None,
    ) -> bool:
        buffers = self.buffers
        if buffers is None:
            return False
        n = int(active_rows)
        if n <= 0 or n > self.rows:
            return False
        previous_active = max(0, min(int(self.active_rows), int(buffers.valid.numel())))
        if previous_active > 0:
            buffers.valid[:previous_active].zero_()
        if not self.external_inputs:
            buffers.pair.src[:n].copy_(src[:n])
            buffers.pair.tgt[:n].copy_(tgt[:n])
        buffers.valid[:n].fill_(1.0)
        slot_buffers = buffers.slots
        if slot_buffers is not None:
            if slot_ids is None:
                return False
            slot_buffers.slot_ids[:n].copy_(slot_ids[:n])
        self.active_rows = n
        return True

    def replay_with_disable_fallback(
        self,
        *,
        src: torch.Tensor,
        tgt: torch.Tensor,
        active_rows: int,
        count_delta: int,
        slot_ids: torch.Tensor | None = None,
        on_replayed: Callable[[torch.Tensor], None],
    ) -> bool:
        if self.capture_state != "captured":
            return False
        self.replay_attempt_count += 1
        if not self.stage_inputs(src=src, tgt=tgt, active_rows=active_rows, slot_ids=slot_ids):
            self.replay_stage_miss_count += 1
            return False
        try:
            with torch.inference_mode():
                assert not isinstance(self.replay_graph, ReplayGraphMissing)
                self.replay_graph.replay()
        except Exception as exc:
            self.capture_state = "disabled"
            self.replay_graph = MISSING_REPLAY_GRAPH
            self.disable_reason = str(exc)
            return False
        buffers = self.buffers
        assert buffers is not None  # stage_inputs succeeded above
        on_replayed(buffers.loss.detach())
        self.replay_hit_count += 1
        return True
