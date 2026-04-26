#!/usr/bin/env python3
"""Unit tests for generation completion trace contracts."""

from __future__ import annotations

import unittest


class _Candidate:
    def __init__(self, token_ids, *, text="", index=None, finish_reason=None):
        self.token_ids = token_ids
        self.text = text
        self.index = index
        self.finish_reason = finish_reason


class _Output:
    def __init__(self, request_id, outputs):
        self.request_id = request_id
        self.outputs = outputs


class _Engine:
    def __init__(self) -> None:
        self.added: list[str] = []

    def add_request(self, request_id, prompt, params, *args, **kwargs):
        self.added.append(str(request_id))


class _LLM:
    def __init__(self, outputs) -> None:
        self.llm_engine = _Engine()
        self._outputs = outputs

    def generate(self, prompts, params):
        for idx, prompt in enumerate(prompts):
            self.llm_engine.add_request(f"req{idx}", prompt, params[idx])
        return self._outputs


class GenerationTraceUnitTest(unittest.TestCase):
    def test_port_catalog_exposes_generation_trace_and_reward_outcome(self) -> None:
        from tllm.ports.base import PortKind
        from tllm.ports.catalog import PUBLIC_PORT_KINDS
        from tllm.ports.generation_trace import GenerationTrace
        from tllm.ports.reward_outcome import RewardOutcome

        self.assertIn(PortKind.GENERATION_TRACE, PUBLIC_PORT_KINDS)
        self.assertIn(PortKind.REWARD_OUTCOME, PUBLIC_PORT_KINDS)
        self.assertEqual(GenerationTrace.read().kind, PortKind.GENERATION_TRACE)
        self.assertEqual(RewardOutcome.write().kind, PortKind.REWARD_OUTCOME)

    def test_build_generation_trace_from_outputs_preserves_prompt_sample_mapping(self) -> None:
        from tllm.runtime.generation_trace_runtime import build_generation_trace_from_outputs

        outputs = [
            _Output(
                "reqA",
                [
                    _Candidate(__import__("torch").tensor([1, 2]), text="alpha", index=0, finish_reason="length"),
                    _Candidate([3], text="beta", index=1, finish_reason="stop"),
                ],
            ),
            _Output("2_reqA", [_Candidate([7, 8], text="gamma")]),
            _Output("bad", [_Candidate([9], text="skip")]),
        ]

        def _resolver(request_id: str) -> tuple[int, int]:
            if request_id == "reqA":
                return 5, 0
            if request_id == "2_reqA":
                return 5, 2
            return -1, -1

        trace = build_generation_trace_from_outputs(outputs, resolve_prompt_sample_fn=_resolver, group_id="vote-5")

        self.assertEqual(len(trace.entries), 3)
        self.assertEqual((trace.entries[0].prompt_idx, trace.entries[0].sample_idx), (5, 0))
        self.assertEqual((trace.entries[1].prompt_idx, trace.entries[1].sample_idx), (5, 1))
        self.assertEqual((trace.entries[2].prompt_idx, trace.entries[2].sample_idx), (5, 2))
        self.assertEqual(trace.entries[0].token_ids, (1, 2))
        self.assertEqual(trace.entries[1].text, "beta")
        self.assertEqual(trace.entries[0].finish_reason, "length")
        self.assertEqual(trace.entries[2].group_id, "vote-5")

    def test_generate_wrapper_stores_last_generation_trace_on_runtime(self) -> None:
        from tllm.runtime.vllm_patch.capture_runner import run_generate_with_request_mapping

        outputs = [_Output("req0", [_Candidate([11], text="done", finish_reason="stop")])]
        runtime = type("Runtime", (), {})()
        llm = _LLM(outputs)

        returned = run_generate_with_request_mapping(
            runtime=runtime,
            llm=llm,
            prompts=["prompt"],
            params=[object()],
            request_prompt_indices=[42],
            request_sample_indices=[3],
            generation_group_id="round-a",
        )

        self.assertIs(returned, outputs)
        self.assertEqual(len(runtime.last_generation_trace.entries), 1)
        entry = runtime.last_generation_trace.entries[0]
        self.assertEqual(entry.request_id, "req0")
        self.assertEqual((entry.prompt_idx, entry.sample_idx), (42, 3))
        self.assertEqual(entry.token_ids, (11,))
        self.assertEqual(entry.group_id, "round-a")

    def test_generate_wrapper_resolves_child_request_ids(self) -> None:
        from tllm.runtime.vllm_patch.capture_runner import run_generate_with_request_mapping

        outputs = [_Output("2_req0", [_Candidate([21], text="child", finish_reason="stop")])]
        runtime = type("Runtime", (), {})()
        llm = _LLM(outputs)

        run_generate_with_request_mapping(
            runtime=runtime,
            llm=llm,
            prompts=["prompt"],
            params=[object()],
            request_prompt_indices=[9],
        )

        self.assertEqual(len(runtime.last_generation_trace.entries), 1)
        entry = runtime.last_generation_trace.entries[0]
        self.assertEqual(entry.request_id, "2_req0")
        self.assertEqual((entry.prompt_idx, entry.sample_idx), (9, 2))


if __name__ == "__main__":
    unittest.main()
