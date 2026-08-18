#!/usr/bin/env python3
"""Toy coverage for GTTA consumer-local dynamics accumulation."""

from __future__ import annotations

import unittest


class _FakeCaptureEngine:
    def __init__(self) -> None:
        self.added_prompts: list[str] = []

    def add_request(self, request_id, prompt, params, *args, **kwargs):
        _ = (request_id, params, args, kwargs)
        self.added_prompts.append(str(prompt))


class _FakeCaptureCandidate:
    token_ids = (1, 2)
    text = "ok"
    finish_reason = "stop"


class _FakeCaptureOutput:
    request_id = "req0"
    outputs = (_FakeCaptureCandidate(),)


class _FakeCaptureLLM:
    def __init__(self) -> None:
        self.llm_engine = _FakeCaptureEngine()
        self.generated_prompts: list[str] = []

    def generate(self, prompts, params):
        self.generated_prompts = [str(prompt) for prompt in prompts]
        for idx, prompt in enumerate(prompts):
            self.llm_engine.add_request(f"req{idx}", prompt, params[idx])
        return [_FakeCaptureOutput()]


class GTTADynamicsToyUnitTest(unittest.TestCase):
    def test_prompt_context_entry_changes_prompt_seen_by_capture_runner(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.runtime.trajectory_store import PromptContextBatch, TrajectoryEvent, write_prompt_contexts
        from tllm.runtime.vllm_patch.capture_runner import run_generate_with_request_mapping

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory_event(
            TrajectoryEvent(
                task_id="maps",
                state_signature="search-results",
                previous_observation="map results",
                action="open directions",
                next_observation="directions panel",
                reward=0.7,
            )
        )
        consumer.process_queued_trajectories()
        context = consumer.build_prompt_context_entry(
            prompt_idx=7,
            task_id="maps",
            state_signature="search-results",
            candidate_actions=("open directions",),
        )

        runtime = type("Runtime", (), {})()
        write_prompt_contexts(runtime, PromptContextBatch(entries=(context,)))
        llm = _FakeCaptureLLM()
        base_prompt = "Observation: map search results\nAction:"

        run_generate_with_request_mapping(
            runtime=runtime,
            llm=llm,
            prompts=[base_prompt],
            params=[object()],
            request_prompt_indices=[7],
            generation_group_id="gtta-smoke",
        )

        injected_prompt = f"{context.text}\n{base_prompt}"
        self.assertNotEqual(injected_prompt, base_prompt)
        self.assertEqual(llm.generated_prompts, [injected_prompt])
        self.assertEqual(llm.llm_engine.added_prompts, [injected_prompt])
        self.assertEqual(runtime.last_generation_trace.entries[0].prompt_idx, 7)
        self.assertEqual(runtime.last_generation_trace.entries[0].group_id, "gtta-smoke")


if __name__ == "__main__":
    unittest.main()
