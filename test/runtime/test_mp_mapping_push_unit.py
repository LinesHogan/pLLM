#!/usr/bin/env python3
"""Unit tests for the request-mapping add_request wrapper (A3)."""

from __future__ import annotations

import unittest

from tllm.runtime.vllm_patch.capture_runner import make_mapping_add_request


class _RecordingPlane:
    def __init__(self) -> None:
        self.pushes: list[list[tuple[str, int, int]]] = []

    def push_request_mapping(self, mappings) -> None:
        self.pushes.append(list(mappings))


class _RuntimeStub:
    reqid_to_promptidx: dict = {}
    reqid_to_sampleidx: dict = {}


class MappingAddRequestUnitTest(unittest.TestCase):
    def test_push_precedes_orig_add_request(self) -> None:
        events: list[str] = []
        plane = _RecordingPlane()

        def orig_add_request(request_id, prompt, p, *args, **kwargs):
            events.append(f"add:{request_id}")
            return request_id

        class Plane(_RecordingPlane):
            def push_request_mapping(self, mappings) -> None:
                events.append("push")
                super().push_request_mapping(mappings)

        plane = Plane()
        runtime = _RuntimeStub()
        wrapped = make_mapping_add_request(
            runtime=runtime,
            plane=plane,
            orig_add_request=orig_add_request,
            request_prompt_indices=[7, 8],
            request_sample_indices=None,
        )
        wrapped("req-a", "prompt", None)
        wrapped("req-b", "prompt", None)
        # Each mapping push must land BEFORE the request is submitted; the
        # FIFO input socket then guarantees the child sees the mapping first.
        self.assertEqual(events, ["push", "add:req-a", "push", "add:req-b"])
        self.assertEqual(plane.pushes, [[("req-a", 7, 0)], [("req-b", 8, 0)]])
        self.assertEqual(runtime.reqid_to_promptidx, {"req-a": 7, "req-b": 8})
        self.assertEqual(runtime.reqid_to_sampleidx, {"req-a": 0, "req-b": 0})

    def test_default_indices_and_reset(self) -> None:
        runtime = _RuntimeStub()
        runtime.reqid_to_promptidx = {"stale": 99}
        wrapped = make_mapping_add_request(
            runtime=runtime,
            plane=None,  # mp mapping push disabled: local record only
            orig_add_request=lambda *a, **k: None,
            request_prompt_indices=None,
            request_sample_indices=None,
        )
        self.assertEqual(runtime.reqid_to_promptidx, {})  # stale state reset
        wrapped("r0", "p", None)
        wrapped("r1", "p", None)
        self.assertEqual(runtime.reqid_to_promptidx, {"r0": 0, "r1": 1})

    def test_sample_indices_used_when_provided(self) -> None:
        plane = _RecordingPlane()
        wrapped = make_mapping_add_request(
            runtime=_RuntimeStub(),
            plane=plane,
            orig_add_request=lambda *a, **k: None,
            request_prompt_indices=None,
            request_sample_indices=[3],
        )
        wrapped("r0", "p", None)
        self.assertEqual(plane.pushes, [[("r0", 0, 3)]])


if __name__ == "__main__":
    unittest.main()
