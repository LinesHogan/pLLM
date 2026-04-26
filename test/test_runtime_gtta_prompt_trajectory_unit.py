#!/usr/bin/env python3
"""Unit tests for GTTA prompt-context and trajectory runtime contracts."""

from __future__ import annotations

import unittest


class _Engine:
    def add_request(self, request_id, prompt, params, *args, **kwargs):
        _ = (request_id, prompt, params, args, kwargs)


class _LLM:
    def __init__(self, outputs) -> None:
        self.llm_engine = _Engine()
        self._outputs = outputs
        self.seen_prompts = None

    def generate(self, prompts, params):
        self.seen_prompts = list(prompts)
        for idx, prompt in enumerate(prompts):
            self.llm_engine.add_request(f"req{idx}", prompt, params[idx])
        return self._outputs


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


class GTTAPromptTrajectoryRuntimeUnitTest(unittest.TestCase):
    def test_trajectory_event_rejects_implicit_coercion(self) -> None:
        from tllm.runtime.dynamics_store import TrajectoryEvent

        with self.assertRaisesRegex(TypeError, "task_id must be str"):
            TrajectoryEvent(task_id=1, state_signature="s", previous_observation="", action="a", next_observation="")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "reward must be real"):
            TrajectoryEvent(task_id="task", state_signature="s", previous_observation="", action="a", next_observation="", reward=True)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "action must be non-empty"):
            TrajectoryEvent(task_id="task", state_signature="s", previous_observation="", action="", next_observation="")

    def test_dynamics_store_keeps_task_state_action_separate_and_copy_safe(self) -> None:
        from tllm.runtime.dynamics_store import DynamicsStore, TrajectoryEvent, TrajectoryEventBatch

        store = DynamicsStore()
        first = TrajectoryEvent(
            task_id="task-a",
            state_signature="state-1",
            previous_observation="obs",
            action="open",
            next_observation="door opened",
            reward=1.0,
        )
        second = TrajectoryEvent(
            task_id="task-a",
            state_signature="state-2",
            previous_observation="obs",
            action="open",
            next_observation="drawer opened",
            reward=0.0,
        )

        self.assertEqual(store.append(TrajectoryEventBatch(entries=(first, second))), 2)
        query = store.query(task_id="task-a", state_signature="state-1", action="open")
        self.assertEqual(query, (first,))
        exported = store.export_text(task_id="task-a", max_records=2)
        self.assertEqual(len(exported), 2)
        self.assertIn("state-1", exported[0])
        self.assertEqual(store.query(task_id="task-a"), (first, second))

    def test_prompt_context_entries_apply_by_prompt_index(self) -> None:
        from tllm.runtime.dynamics_store import (
            PromptContextBatch,
            PromptContextEntry,
            apply_prompt_contexts,
            write_prompt_contexts,
        )

        runtime = type("Runtime", (), {})()
        batch = PromptContextBatch(
            entries=(
                PromptContextEntry(prompt_idx=3, text="[ctx-a]", position="prefix"),
                PromptContextEntry(prompt_idx=4, text="[ctx-b]", position="suffix"),
            )
        )
        write_prompt_contexts(runtime, batch)

        prompts = apply_prompt_contexts(runtime, ("prompt a", "prompt b"), request_prompt_indices=(3, 4))

        self.assertEqual(prompts, ("[ctx-a]\nprompt a", "prompt b\n[ctx-b]"))

    def test_prompt_context_rejects_implicit_types(self) -> None:
        from tllm.runtime.dynamics_store import PromptContextEntry

        with self.assertRaisesRegex(TypeError, "prompt_idx must be int"):
            PromptContextEntry(prompt_idx=True, text="ctx")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "text must be str"):
            PromptContextEntry(prompt_idx=0, text=3)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "unsupported prompt context position"):
            PromptContextEntry(prompt_idx=0, text="ctx", position="middle")

    def test_generate_wrapper_applies_prompt_context_before_llm_generate(self) -> None:
        from tllm.runtime.dynamics_store import PromptContextBatch, PromptContextEntry, write_prompt_contexts
        from tllm.runtime.vllm_patch.capture_runner import run_generate_with_request_mapping

        runtime = type("Runtime", (), {})()
        write_prompt_contexts(
            runtime,
            PromptContextBatch(entries=(PromptContextEntry(prompt_idx=9, text="state memory", position="prefix"),)),
        )
        llm = _LLM([_Output("req0", [_Candidate([1], text="ok", finish_reason="stop")])])

        run_generate_with_request_mapping(
            runtime=runtime,
            llm=llm,
            prompts=["base prompt"],
            params=[object()],
            request_prompt_indices=[9],
        )

        self.assertEqual(llm.seen_prompts, ["state memory\nbase prompt"])
        self.assertEqual(len(runtime.last_generation_trace.entries), 1)


if __name__ == "__main__":
    unittest.main()
