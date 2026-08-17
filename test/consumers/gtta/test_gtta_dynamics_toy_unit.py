#!/usr/bin/env python3
"""Toy coverage for GTTA consumer-local dynamics accumulation."""

from __future__ import annotations

import json
import os
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
    def test_runtime_trajectory_event_uses_explicit_task_state_action_key(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.runtime.trajectory_store import TrajectoryEvent

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory_event(
            TrajectoryEvent(
                task_id="checkout",
                state_signature="cart-ready",
                previous_observation="same rendered page text",
                action="click submit",
                next_observation="order confirmation",
                reward=1.0,
            )
        )
        consumer.enqueue_trajectory_event(
            TrajectoryEvent(
                task_id="checkout",
                state_signature="cart-empty",
                previous_observation="same rendered page text",
                action="click submit",
                next_observation="validation error",
                reward=-1.0,
            )
        )
        consumer.enqueue_trajectory_event(
            TrajectoryEvent(
                task_id="cancel",
                state_signature="cart-ready",
                previous_observation="same rendered page text",
                action="click submit",
                next_observation="wrong workflow",
                reward=-0.5,
            )
        )

        consumer.process_queued_trajectories()

        self.assertGreater(
            consumer.score_action("click submit", task_id="checkout", state_signature="cart-ready"),
            0.0,
        )
        self.assertLess(
            consumer.score_action("click submit", task_id="checkout", state_signature="cart-empty"),
            0.0,
        )
        self.assertLess(
            consumer.score_action("click submit", task_id="cancel", state_signature="cart-ready"),
            0.0,
        )
        records = consumer.query_dynamics_records(
            task_id="checkout",
            state_signature="cart-ready",
            action="click submit",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["previous_observation"], "same rendered page text")
        self.assertEqual(records[0]["next_observation"], "order confirmation")

    def test_enqueue_trajectory_event_rejects_implicit_shapes(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig

        consumer = GTTAConsumer(GTTAConsumerConfig())

        with self.assertRaisesRegex(TypeError, "TrajectoryEvent"):
            consumer.enqueue_trajectory_event(
                {
                    "task_id": "checkout",
                    "state_signature": "cart-ready",
                    "previous_observation": "page",
                    "action": "click submit",
                    "next_observation": "done",
                    "reward": 1.0,
                }
            )

    def test_query_dynamics_records_filters_task_state_and_action(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="reserve_table",
                action="click confirm",
                previous_observation="booking form with a 7 pm time selected",
                next_observation="reservation confirmation page",
                environmental_dynamics="The restaurant accepted the selected time.",
                reward=1.0,
            )
        )
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="reserve_table",
                action="click confirm",
                previous_observation="booking form with no time selected",
                next_observation="validation error asking for a time",
                environmental_dynamics="Confirmation failed because the time field was empty.",
                reward=-0.5,
            )
        )
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="cancel_table",
                action="click confirm",
                previous_observation="booking form with a 7 pm time selected",
                next_observation="wrong workflow page",
                environmental_dynamics="The same button is not useful for the cancel task.",
                reward=-1.0,
            )
        )

        consumer.process_queued_trajectories()

        matching_records = consumer.query_dynamics_records(
            task_id="reserve_table",
            observation="booking form with a 7 pm time selected",
            action="click confirm",
        )
        ranked = consumer.rank_candidate_actions(
            task_id="reserve_table",
            observation="booking form with a 7 pm time selected",
            candidate_actions=["click confirm", "go back"],
        )

        self.assertEqual(len(matching_records), 1)
        self.assertEqual(
            matching_records[0]["environmental_dynamics"],
            "The restaurant accepted the selected time.",
        )
        self.assertEqual(matching_records[0]["reward"], 1.0)
        self.assertEqual([action for action, _score in ranked], ["click confirm", "go back"])

    def test_query_dynamics_records_returns_copy(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="wishlist",
                action="click heart",
                previous_observation="product page",
                next_observation="wishlist counter incremented",
                environmental_dynamics="The item was added to the wishlist.",
                reward=0.8,
            )
        )
        consumer.process_queued_trajectories()

        queried = consumer.query_dynamics_records(task_id="wishlist")
        queried[0]["environmental_dynamics"] = "mutated by caller"

        self.assertEqual(
            consumer.query_dynamics_records(task_id="wishlist")[0]["environmental_dynamics"],
            "The item was added to the wishlist.",
        )

    def test_dynamics_snippets_and_prompt_context_entry_are_deterministic(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.runtime.trajectory_store import PromptContextEntry, TrajectoryEvent

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
        consumer.enqueue_trajectory_event(
            TrajectoryEvent(
                task_id="maps",
                state_signature="search-results",
                previous_observation="map results",
                action="clear search",
                next_observation="blank map",
                reward=-0.2,
            )
        )
        consumer.process_queued_trajectories()

        snippets = consumer.render_ranked_dynamics_snippets(
            task_id="maps",
            state_signature="search-results",
            candidate_actions=("clear search", "open directions"),
        )
        context = consumer.build_prompt_context_entry(
            prompt_idx=5,
            task_id="maps",
            state_signature="search-results",
            candidate_actions=("clear search", "open directions"),
        )

        self.assertEqual(
            snippets,
            (
                "task=maps state=search-results action=open directions reward=0.7 next=directions panel",
                "task=maps state=search-results action=clear search reward=-0.2 next=blank map",
            ),
        )
        self.assertIsInstance(context, PromptContextEntry)
        self.assertEqual(context.prompt_idx, 5)
        self.assertEqual(context.position, "prefix")
        self.assertEqual(context.text, "\n".join(snippets))

    def test_environment_dynamics_context_uses_upstream_shaped_jsonl(self) -> None:
        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.consumers.gtta import TrajectoryFeedback

        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="shopping",
                action="click add to cart",
                previous_observation="Product page with cart button.",
                next_observation="Cart count increased.",
                environmental_dynamics="The selected product is added to the cart and the cart count increments.",
                reward=0.8,
            )
        )
        consumer.enqueue_trajectory(
            TrajectoryFeedback(
                task_id="shopping",
                action="click disabled checkout",
                previous_observation="Product page with cart button.",
                next_observation="No visible change.",
                environmental_dynamics="No checkout happened because the cart was not opened.",
                reward=-0.1,
            )
        )
        consumer.process_queued_trajectories()

        context_text = consumer.render_environment_dynamics_context(
            task_id="shopping",
            observation="Product page with cart button.",
            candidate_actions=("click disabled checkout", "click add to cart"),
        )
        lines = [json.loads(line) for line in context_text.splitlines()]

        self.assertEqual(
            [line["action_taken"] for line in lines],
            ["click add to cart", "click disabled checkout"],
        )
        self.assertEqual(lines[0]["initial_state"], "Product page with cart button.")
        self.assertEqual(
            lines[0]["environmental_dynamics"],
            "The selected product is added to the cart and the cart count increments.",
        )
        self.assertEqual(lines[0]["task_id"], "shopping")
        self.assertEqual(lines[0]["reward"], 0.8)

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


@unittest.skipUnless(
    os.environ.get("TLLM_GTTA_RUN_CACHED_MODEL_SMOKE") == "1",
    "set TLLM_GTTA_RUN_CACHED_MODEL_SMOKE=1 to run local cached-model smoke",
)
class GTTACachedModelSmokeTest(unittest.TestCase):
    def test_prompt_context_changes_tiny_gpt2_logits(self) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from tllm.consumers.gtta import GTTAConsumer, GTTAConsumerConfig
        from tllm.runtime.trajectory_store import TrajectoryEvent, apply_prompt_contexts

        model_id = "sshleifer/tiny-gpt2"
        consumer = GTTAConsumer(GTTAConsumerConfig())
        consumer.enqueue_trajectory_event(
            TrajectoryEvent(
                task_id="maps",
                state_signature="search-results",
                previous_observation="OpenStreetMap page with search results",
                action="open directions",
                next_observation="directions panel opened",
                reward=0.7,
            )
        )
        consumer.process_queued_trajectories()

        runtime = type("Runtime", (), {})()
        context = consumer.build_prompt_context_entry(
            prompt_idx=0,
            task_id="maps",
            state_signature="search-results",
            candidate_actions=("open directions",),
        )
        from tllm.runtime.trajectory_store import PromptContextBatch, write_prompt_contexts

        write_prompt_contexts(runtime, PromptContextBatch(entries=(context,)))
        prompt = "Task: choose the next browser action. Observation: map search results. Action:"
        injected_prompt = apply_prompt_contexts(runtime, (prompt,))[0]

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True)
        model.eval()
        with torch.no_grad():
            base_logits = model(**tokenizer(prompt, return_tensors="pt")).logits[0, -1]
            injected_logits = model(**tokenizer(injected_prompt, return_tensors="pt")).logits[0, -1]

        self.assertGreater(torch.max(torch.abs(base_logits - injected_logits)).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
