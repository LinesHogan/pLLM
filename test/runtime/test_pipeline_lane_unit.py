#!/usr/bin/env python3
"""CPU contracts for shared adaptation pipeline event ordering."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.runtime.scheduling.streams import PipelineBuffers, PipelineLane, StreamPool


class _Event:
    def __init__(self, name: str, trace: list[str]) -> None:
        self.name = name
        self.trace = trace

    def record(self, stream) -> None:
        self.trace.append(f"record:{self.name}")


class _Stream:
    def __init__(self, trace: list[str]) -> None:
        self.trace = trace

    def wait_event(self, event: _Event) -> None:
        self.trace.append(f"wait:{event.name}")


class PipelineLaneUnitTest(unittest.TestCase):
    def test_next_forward_waits_for_staging_but_not_training(self) -> None:
        trace: list[str] = []
        lane = PipelineLane(streams=StreamPool(mode="serial"), slots=1, label="test")
        lane.buffers = PipelineBuffers(
            src=torch.empty((1, 1, 1)),
            tgt=torch.empty((1, 1, 1)),
        )
        lane.tgt_done_events = [_Event("tgt", trace)]  # type: ignore[list-item]
        lane.slot_train_done_events = [_Event("train", trace)]  # type: ignore[list-item]
        lane.pending_reuse_slots.add(0)

        with mock.patch("torch.cuda.current_stream", return_value=_Stream(trace)):
            lane.retire_step_inputs()

        self.assertEqual(trace, ["wait:tgt"])

    def test_parameter_reader_waits_for_latest_training_publish(self) -> None:
        trace: list[str] = []
        lane = PipelineLane(streams=StreamPool(mode="serial"), slots=1, label="test")
        lane.buffers = PipelineBuffers(
            src=torch.empty((1, 1, 1), device="meta"),
            tgt=torch.empty((1, 1, 1), device="meta"),
        )
        lane.slot_train_done_events = [_Event("train", trace)]  # type: ignore[list-item]
        lane.pending_parameter_slots.add(0)

        # A CUDA-like fake buffer device is not constructible in CPU-only CI;
        # pass the reader stream and patch the layout check at this boundary.
        lane.buffers = mock.Mock(src=mock.Mock(device=torch.device("cuda")))
        lane.wait_for_pending_parameters(stream=_Stream(trace))  # type: ignore[arg-type]

        self.assertEqual(trace, ["wait:train"])
        self.assertEqual(lane.pending_parameter_slots, set())

    def test_training_waits_for_external_parameter_reader(self) -> None:
        trace: list[str] = []
        lane = PipelineLane(streams=StreamPool(mode="serial"), slots=1, label="test")
        lane.buffers = PipelineBuffers(
            src=torch.empty((1, 1, 1)),
            tgt=torch.empty((1, 1, 1)),
        )
        lane.ready_step_slot = 0
        lane.src_staged_events = [_Event("src", trace)]  # type: ignore[list-item]
        lane.tgt_done_events = [_Event("tgt", trace)]  # type: ignore[list-item]
        lane.slot_train_done_events = [_Event("train", trace)]  # type: ignore[list-item]
        parameter_read = _Event("parameter_read", trace)

        with mock.patch("torch.cuda.current_stream", return_value=_Stream(trace)):
            lane.enqueue_ready_and_drain(
                active_rows=1,
                prompt_idxs=(7,),
                train_slot=lambda slot, rows, prompts: trace.append("train"),
                wait_events=(parameter_read,),
            )

        self.assertEqual(
            trace,
            [
                "wait:src",
                "wait:tgt",
                "wait:parameter_read",
                "train",
                "record:train",
            ],
        )


if __name__ == "__main__":
    unittest.main()
