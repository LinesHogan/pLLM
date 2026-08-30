#!/usr/bin/env python3
"""Unit tests for the request-mapping add_request wrapper (A3)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from tllm.runtime.vllm_patch.capture_runner import (
    make_mapping_add_request,
    run_generate_with_request_mapping,
)
from tllm.runtime.vllm_adapter import install_generate_boundary
from tllm.runtime.request_lifecycle import RequestLifecycleLedger
from tllm.contracts.request_identity import RequestIdentity


class _RecordingPlane:
    def __init__(self) -> None:
        self.pushes: list[list[tuple[str, int, int | None]]] = []

    def push_request_mapping(self, mappings) -> None:
        self.pushes.append(list(mappings))


class _RuntimeStub:
    def __init__(self) -> None:
        self.reqid_to_promptidx: dict[str, int] = {}
        self.reqid_to_sampleidx: dict[str, int] = {}
        self.request_lifecycle = RequestLifecycleLedger()


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
        self.assertEqual(plane.pushes, [[("req-a", 7, None)], [("req-b", 8, None)]])
        self.assertEqual(runtime.reqid_to_promptidx, {"req-a": 7, "req-b": 8})
        self.assertEqual(runtime.reqid_to_sampleidx, {})
        self.assertEqual(
            tuple(alias.identity for alias in runtime.request_lifecycle.active_aliases),
            (RequestIdentity("req-a", 7, 0), RequestIdentity("req-b", 8, 0)),
        )

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

    def test_generate_wrapper_forwards_progress_setting(self) -> None:
        calls: list[dict[str, object]] = []

        class Engine:
            add_request = staticmethod(lambda *args, **kwargs: None)

        class LLM:
            llm_engine = Engine()

            def generate(self, prompts, params, **kwargs):
                calls.append(dict(kwargs))
                return []

        runtime = _RuntimeStub()
        with (
            patch("tllm.runtime.mp.control.ControlPlane.try_bind", return_value=None),
            patch("tllm.runtime.vllm_patch.capture_runner.apply_prompt_contexts", side_effect=lambda _r, prompts, **_k: prompts),
            patch("tllm.runtime.vllm_patch.capture_runner.build_generation_trace_from_outputs", return_value=None),
            patch("tllm.runtime.vllm_patch.capture_runner.dispatch_post_generate_port_bundles"),
        ):
            run_generate_with_request_mapping(
                runtime=runtime,
                llm=LLM(),
                prompts=["p"],
                params=object(),
                use_tqdm=False,
            )

        self.assertEqual(calls, [{"use_tqdm": False}])

    def test_boundary_aborts_actual_admission_after_nonprepublished_failure(self) -> None:
        from tllm.runtime import residual_runtime
        from tllm.runtime.state import ResidualRuntimeState

        runtime = _RuntimeStub()
        state = ResidualRuntimeState()
        aborted: list[tuple[str, ...]] = []

        class Engine:
            def add_request(self, request_id, _prompt, _params):
                return request_id

        class LLM:
            def __init__(self) -> None:
                self.llm_engine = Engine()

            def generate(self, _prompts, _params):
                self.llm_engine.add_request("request-0", "prompt", None)
                raise RuntimeError("engine failed after admission")

        llm = LLM()
        with (
            patch("tllm.runtime.mp.control.ControlPlane.try_bind", return_value=None),
            patch.object(residual_runtime, "RUNTIME", state),
        ):
            install_generate_boundary(
                llm,
                runtime,
                lambda: None,
                lambda _outputs: None,
                lambda request_ids: aborted.append(request_ids),
                lambda: (7,),
                lambda: None,
            )
            with self.assertRaisesRegex(RuntimeError, "engine failed after admission"):
                llm.generate(("prompt",), None)

        self.assertEqual(aborted, [("request-0",)])
        self.assertEqual(state.reqid_to_promptidx, {"request-0": 7})
        self.assertEqual(len(state.request_lifecycle.active_aliases), 1)


if __name__ == "__main__":
    unittest.main()
