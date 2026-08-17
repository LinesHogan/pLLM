#!/usr/bin/env python3
"""Unit tests for the FTTT-inspired consumer."""

from __future__ import annotations

import sys
import unittest

import torch


class FTTTConsumerUnitTest(unittest.TestCase):
    def test_package_exports_fttt_consumer(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig

        consumer = FTTTConsumer(FTTTConsumerConfig())

        self.assertEqual(consumer.consumer_id, "fttt")

    def test_enqueue_feedback_records_bounded_queue(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(max_queue_size=1))

        accepted = consumer.enqueue_feedback(FeedbackRecord(prompt_idx=0, feedback="incorrect", reward=0.0))
        dropped = consumer.enqueue_feedback(FeedbackRecord(prompt_idx=1, feedback="incorrect", reward=0.0))

        stats = consumer.read_stats()
        self.assertTrue(accepted)
        self.assertFalse(dropped)
        self.assertEqual(stats.queued_feedback, 1)
        self.assertEqual(stats.dropped_feedback, 1)

    def test_apply_feedback_update_changes_adapter_and_records_stats(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        hidden = torch.randn(2, 4)
        before = consumer.engine.adapter.up.weight.detach().clone()

        consumer.apply_feedback_update(
            hidden,
            [
                FeedbackRecord(prompt_idx=0, feedback="incorrect", reward=0.0),
                FeedbackRecord(prompt_idx=1, feedback="correct", reward=1.0),
            ],
        )

        after = consumer.engine.adapter.up.weight.detach()
        stats = consumer.read_stats()
        self.assertFalse(torch.allclose(before, after))
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.feedback_count, 1)
        self.assertEqual(stats.skipped_feedback, 1)

    def test_same_reward_with_different_feedback_text_produces_different_edit(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.fttt import FeedbackRecord

        torch.manual_seed(7)
        config = FTTTConsumerConfig(hidden_size=4, rank=3, learning_rate=0.2)
        hidden = torch.tensor([[0.25, -0.5, 0.75, 1.0]])

        arithmetic = FTTTConsumer(config)
        arithmetic_feedback = [FeedbackRecord(prompt_idx=0, feedback="wrong arithmetic carry", reward=0.0)]
        arithmetic.apply_feedback_update(hidden, arithmetic_feedback)
        arithmetic_delta = arithmetic.edit_delta(hidden, arithmetic_feedback)

        torch.manual_seed(7)
        notation = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=3, learning_rate=0.2))
        notation_feedback = [FeedbackRecord(prompt_idx=0, feedback="wrong final notation", reward=0.0)]
        notation.apply_feedback_update(hidden, notation_feedback)
        notation_delta = notation.edit_delta(hidden, notation_feedback)

        self.assertFalse(torch.allclose(arithmetic_delta, notation_delta))

    def test_positive_feedback_stops_later_adaptation_for_instance(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        hidden = torch.randn(1, 4)

        consumer.apply_feedback_update(hidden, [FeedbackRecord(prompt_idx=3, feedback="correct", reward=1.0)])
        consumer.apply_feedback_update(hidden, [FeedbackRecord(prompt_idx=3, feedback="now fix arithmetic", reward=0.0)])

        state = consumer.instance_state(3)
        stats = consumer.read_stats()
        self.assertTrue(state.stopped)
        self.assertEqual(state.retries, 0)
        self.assertEqual(stats.update_count, 0)
        self.assertEqual(stats.skipped_feedback, 2)

    def test_per_instance_reset_allows_adaptation_after_stop(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        hidden = torch.randn(1, 4)

        consumer.apply_feedback_update(hidden, [FeedbackRecord(prompt_idx=3, feedback="correct", reward=1.0)])
        consumer.reset_instance(3)
        consumer.apply_feedback_update(hidden, [FeedbackRecord(prompt_idx=3, feedback="fix the sign", reward=0.0)])

        state = consumer.instance_state(3)
        stats = consumer.read_stats()
        self.assertFalse(state.stopped)
        self.assertEqual(state.retries, 1)
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.feedback_count, 1)

    def test_edit_api_changes_hidden_and_logits(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.2))
        hidden = torch.randn(1, 4)
        feedback = [FeedbackRecord(prompt_idx=0, feedback="revise the last operation", reward=0.0)]
        consumer.apply_feedback_update(hidden, feedback)

        edited_hidden = consumer.apply_edit_to_hidden(hidden, feedback)
        logits = torch.zeros(1, 4)
        edited_logits = consumer.apply_edit_to_logits(logits, hidden, feedback)

        self.assertFalse(torch.allclose(edited_hidden, hidden))
        self.assertFalse(torch.allclose(edited_logits, logits))

    def test_inference_mode_hidden_is_accepted(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        with torch.inference_mode():
            hidden = torch.randn(1, 4)

        consumer.apply_feedback_update(hidden, [FeedbackRecord(prompt_idx=0, feedback="incorrect", reward=0.0)])

        self.assertEqual(consumer.read_stats().update_count, 1)

    def test_disabled_consumer_does_not_update(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.fttt import FeedbackRecord

        consumer = FTTTConsumer(FTTTConsumerConfig(enabled=False, hidden_size=4))

        consumer.apply_feedback_update(torch.randn(1, 4), [FeedbackRecord(prompt_idx=0, feedback="incorrect", reward=0.0)])

        self.assertEqual(consumer.read_stats().update_count, 0)

    def test_trace_failure_is_judged_queued_and_returned_as_reward_outcome(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
        from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch

        judged: list[tuple[str, str]] = []

        def judge(entry: CompletionTraceEntry) -> FTTTJudgeResult:
            judged.append((entry.request_id, entry.text))
            return FTTTJudgeResult(
                is_correct=False,
                reward=0.0,
                feedback_text="Your answer is incorrect. Recheck the arithmetic.",
                label="expected=4",
                judge="toy-arithmetic",
            )

        consumer = FTTTConsumer(FTTTConsumerConfig(), judge=judge)
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-fail",
                    prompt_idx=7,
                    sample_idx=2,
                    output_index=1,
                    token_ids=(11, 12),
                    text="5",
                    finish_reason="stop",
                    group_id="round-1",
                ),
            )
        )

        batch = consumer.process_generation_trace(trace)

        self.assertIsInstance(batch, RewardOutcomeBatch)
        self.assertEqual(judged, [("req-fail", "5")])
        self.assertEqual(len(batch.entries), 1)
        outcome = batch.entries[0]
        self.assertEqual(outcome.request_id, "req-fail")
        self.assertEqual(outcome.prompt_idx, 7)
        self.assertEqual(outcome.sample_idx, 2)
        self.assertEqual(outcome.output_index, 1)
        self.assertEqual(outcome.group_id, "round-1")
        self.assertEqual(outcome.reward, 0.0)
        self.assertEqual(outcome.judge, "toy-arithmetic")
        self.assertEqual(outcome.label, "expected=4")
        self.assertEqual(outcome.feedback_text, "Your answer is incorrect. Recheck the arithmetic.")
        self.assertEqual(outcome.skip_reason, "")
        self.assertFalse(outcome.is_correct)

        queued = consumer.drain_feedback()
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0].prompt_idx, 7)
        self.assertEqual(queued[0].feedback, "Your answer is incorrect. Recheck the arithmetic.")
        self.assertEqual(queued[0].reward, 0.0)

    def test_consume_bundle_writes_reward_outcome_sink(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.contracts.port_bundle import BundleKey, PortBundle
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
        from tllm.runtime.reward_outcome_runtime import RewardOutcomeBatch

        class _Sink:
            def __init__(self) -> None:
                self.batches: list[RewardOutcomeBatch] = []

            def append(self, batch: RewardOutcomeBatch) -> int:
                self.batches.append(batch)
                return len(batch.entries)

        sink = _Sink()
        consumer = FTTTConsumer(
            FTTTConsumerConfig(),
            judge=lambda entry: FTTTJudgeResult(
                is_correct=False,
                reward=0.0,
                feedback_text=f"fix {entry.text}",
                judge="toy",
            ),
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-fail",
                    prompt_idx=7,
                    sample_idx=2,
                    output_index=0,
                    token_ids=(),
                    text="5",
                    finish_reason="stop",
                    group_id="round-1",
                ),
            )
        )
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="post_generate", request_id="", sample_idx=0),
            entries={"generation_trace": trace, "reward_outcome": sink},
        )

        consumer.consume_bundle(bundle, ctx=object())

        self.assertEqual(len(sink.batches), 1)
        self.assertEqual(sink.batches[0].entries[0].feedback_text, "fix 5")
        self.assertEqual(consumer.read_stats().queued_feedback, 1)

    def test_consume_bundle_writes_closed_loop_retry_and_update_barrier_sinks(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.contracts.port_bundle import BundleKey, PortBundle
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        class _Sink:
            def __init__(self) -> None:
                self.batches: list[object] = []

            def append(self, batch: object) -> int:
                self.batches.append(batch)
                return len(batch.entries)

        reward_sink = _Sink()
        retry_sink = _Sink()
        barrier_sink = _Sink()
        consumer = FTTTConsumer(
            FTTTConsumerConfig(max_retries=2),
            judge=lambda entry: FTTTJudgeResult(
                is_correct=False,
                reward=0.0,
                feedback_text="Your answer is incorrect.",
                judge="toy",
            ),
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-fail",
                    prompt_idx=5,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(),
                    text="wrong",
                    finish_reason="stop",
                    group_id="round-a",
                ),
            )
        )
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="post_generate", request_id="", sample_idx=0),
            entries={
                "generation_trace": trace,
                "reward_outcome": reward_sink,
                "retry_control": retry_sink,
                "update_barrier": barrier_sink,
            },
        )

        consumer.consume_bundle(bundle, ctx=object())

        self.assertEqual(reward_sink.batches[0].entries[0].reward, 0.0)
        self.assertEqual(retry_sink.batches[0].entries[0].action, "retry")
        self.assertTrue(retry_sink.batches[0].entries[0].retry_after_update)
        self.assertEqual(barrier_sink.batches[0].entries[0].state, "ready")
        self.assertEqual(barrier_sink.batches[0].entries[0].ready_count, 1)
        self.assertEqual(consumer.read_stats().queued_feedback, 1)

    def test_trace_success_is_judged_and_not_queued_for_feedback(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        consumer = FTTTConsumer(
            FTTTConsumerConfig(),
            judge=lambda entry: FTTTJudgeResult(
                is_correct=True,
                reward=1.0,
                feedback_text="correct final answer",
                label="42",
                judge="toy-exact-match",
            ),
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-ok",
                    prompt_idx=1,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(4, 2),
                    text="42",
                    finish_reason="stop",
                    group_id="round-ok",
                ),
            )
        )

        batch = consumer.process_generation_trace(trace)

        self.assertEqual(consumer.drain_feedback(), [])
        self.assertEqual(len(batch.entries), 1)
        outcome = batch.entries[0]
        self.assertTrue(outcome.is_correct)
        self.assertEqual(outcome.reward, 1.0)
        self.assertEqual(outcome.feedback_text, "correct final answer")
        self.assertEqual(outcome.label, "42")
        self.assertEqual(outcome.skip_reason, "correct-output")

    def test_closed_loop_intent_turns_failed_completion_into_feedback_retry_and_barrier(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        consumer = FTTTConsumer(
            FTTTConsumerConfig(max_retries=2),
            judge=lambda entry: FTTTJudgeResult(
                is_correct=False,
                reward=0.0,
                feedback_text=f"retry with corrected arithmetic for {entry.text}",
                label="expected=4",
                judge="toy-arithmetic",
            ),
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-fail",
                    prompt_idx=9,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(),
                    text="5",
                    finish_reason="stop",
                    group_id="round-1",
                ),
            )
        )

        intent = consumer.closed_loop_intent_for_trace(trace)

        self.assertEqual(intent.reward_outcomes.entries[0].feedback_text, "retry with corrected arithmetic for 5")
        self.assertEqual(intent.feedback_records[0].feedback, "retry with corrected arithmetic for 5")
        self.assertEqual(intent.retry_controls.entries[0].action, "retry")
        self.assertEqual(intent.retry_controls.entries[0].reason, "feedback-update")
        self.assertTrue(intent.retry_controls.entries[0].retry_after_update)
        barrier = intent.update_barriers.entries[0]
        self.assertEqual(barrier.barrier_id, "fttt:round-1:9:0")
        self.assertEqual(barrier.group_id, "round-1")
        self.assertEqual(barrier.consumer_id, "fttt")
        self.assertEqual(barrier.state, "ready")
        self.assertEqual(barrier.ready_count, 1)
        self.assertEqual(intent.update_count, 1)

    def test_closed_loop_intent_stops_retry_after_instance_lifecycle_reaches_limit(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        consumer = FTTTConsumer(
            FTTTConsumerConfig(max_retries=1),
            judge=lambda entry: FTTTJudgeResult(is_correct=False, reward=0.0, feedback_text="fix final answer"),
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-fail",
                    prompt_idx=3,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(),
                    text="bad",
                    finish_reason="stop",
                    group_id="round-stop",
                ),
            )
        )

        intent = consumer.closed_loop_intent_for_trace(trace)
        consumer.apply_feedback_update(torch.randn(intent.update_count, 4), intent.feedback_records)

        state = consumer.instance_state(3)
        stats = consumer.read_stats()
        self.assertEqual(intent.retry_controls.entries[0].action, "stop")
        self.assertEqual(intent.retry_controls.entries[0].reason, "retry-limit")
        self.assertTrue(state.stopped)
        self.assertEqual(state.retries, 1)
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.feedback_count, 1)

    def test_trace_derived_retry_lifecycle_is_scoped_by_request_id(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        consumer = FTTTConsumer(
            FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1, max_retries=2),
            judge=lambda entry: FTTTJudgeResult(is_correct=False, reward=0.0, feedback_text=f"fix {entry.request_id}"),
        )

        first_trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-a",
                    prompt_idx=0,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(),
                    text="bad-a",
                    finish_reason="stop",
                    group_id="round-a",
                ),
            )
        )
        first_intent = consumer.closed_loop_intent_for_trace(first_trace)
        consumer.apply_feedback_update(torch.randn(first_intent.update_count, 4), first_intent.feedback_records)

        second_trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-b",
                    prompt_idx=0,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(),
                    text="bad-b",
                    finish_reason="stop",
                    group_id="round-b",
                ),
            )
        )

        second_intent = consumer.closed_loop_intent_for_trace(second_trace)

        self.assertEqual(first_intent.retry_controls.entries[0].action, "retry")
        self.assertEqual(second_intent.retry_controls.entries[0].action, "retry")
        self.assertTrue(second_intent.retry_controls.entries[0].retry_after_update)
        self.assertEqual(second_intent.update_barriers.entries[0].state, "ready")
        self.assertEqual(second_intent.update_count, 1)

    def test_closed_loop_intent_simulates_batch_retry_progression_per_instance(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        consumer = FTTTConsumer(
            FTTTConsumerConfig(max_retries=2),
            judge=lambda entry: FTTTJudgeResult(is_correct=False, reward=0.0, feedback_text=f"fix {entry.output_index}"),
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-same",
                    prompt_idx=0,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(),
                    text="bad-0",
                    finish_reason="stop",
                    group_id="round-same",
                ),
                CompletionTraceEntry(
                    request_id="req-same",
                    prompt_idx=0,
                    sample_idx=1,
                    output_index=1,
                    token_ids=(),
                    text="bad-1",
                    finish_reason="stop",
                    group_id="round-same",
                ),
            )
        )

        intent = consumer.closed_loop_intent_for_trace(trace)

        self.assertEqual([entry.action for entry in intent.retry_controls.entries], ["retry", "stop"])
        self.assertEqual([entry.retry_after_update for entry in intent.retry_controls.entries], [True, False])
        self.assertEqual([entry.state for entry in intent.update_barriers.entries], ["ready", "ready"])
        self.assertEqual(intent.update_count, 2)

    def test_closed_loop_intent_blocks_barrier_for_stopped_instance(self) -> None:
        from tllm.consumers.fttt import FeedbackRecord, FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        consumer = FTTTConsumer(
            FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1, max_retries=3),
            judge=lambda entry: FTTTJudgeResult(is_correct=False, reward=0.0, feedback_text="fix stopped"),
        )
        consumer.apply_feedback_update(
            torch.randn(1, 4),
            [FeedbackRecord(prompt_idx=2, feedback="correct", reward=1.0, request_id="req-stopped")],
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-stopped",
                    prompt_idx=2,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(),
                    text="bad",
                    finish_reason="stop",
                    group_id="round-stopped",
                ),
            )
        )

        intent = consumer.closed_loop_intent_for_trace(trace)

        self.assertEqual(intent.feedback_records, ())
        self.assertEqual(intent.update_count, 0)
        self.assertEqual(intent.retry_controls.entries[0].action, "stop")
        self.assertFalse(intent.retry_controls.entries[0].retry_after_update)
        barrier = intent.update_barriers.entries[0]
        self.assertEqual(barrier.state, "blocked")
        self.assertEqual(barrier.ready_count, 0)

    def test_toy_closed_loop_uses_failed_completion_context_in_adapter_update(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        def run_once(failed_text: str) -> torch.Tensor:
            torch.manual_seed(31)
            consumer = FTTTConsumer(
                FTTTConsumerConfig(hidden_size=4, feedback_feature_size=5, rank=2, learning_rate=0.2, max_retries=3),
                judge=lambda entry: FTTTJudgeResult(
                    is_correct=False,
                    reward=0.0,
                    feedback_text="Your answer is incorrect. Recheck the arithmetic.",
                    feedback_token_ids=(101, 202, 303),
                ),
            )
            trace = GenerationTraceBatch(
                entries=(
                    CompletionTraceEntry(
                        request_id="req-fail",
                        prompt_idx=4,
                        sample_idx=0,
                        output_index=0,
                        token_ids=(9, 9),
                        text=failed_text,
                        finish_reason="stop",
                        group_id="round-context",
                    ),
                )
            )

            result = consumer.run_toy_closed_loop_step(trace, torch.eye(1, 4))

            self.assertEqual(result.feedback_records[0].failed_text, failed_text)
            self.assertEqual(result.feedback_records[0].failed_token_ids, (9, 9))
            self.assertEqual(result.feedback_records[0].feedback_token_ids, (101, 202, 303))
            self.assertEqual(result.adapter_version, 1)
            self.assertEqual(result.instance_states[4].retries, 1)
            self.assertEqual(result.retry_controls.entries[0].action, "retry")
            self.assertIn("adapter-version=1", result.retry_controls.entries[0].reason)
            self.assertEqual(consumer.read_stats().update_count, 1)
            return consumer.engine.adapter.up.weight.detach().clone()

        update_from_5 = run_once("5")
        update_from_7 = run_once("7")

        self.assertFalse(torch.allclose(update_from_5, update_from_7))

    def test_toy_closed_loop_feedback_token_ids_drive_adapter_update(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch

        def run_once(feedback_token_ids: tuple[int, ...]) -> torch.Tensor:
            torch.manual_seed(41)
            consumer = FTTTConsumer(
                FTTTConsumerConfig(hidden_size=4, feedback_feature_size=5, rank=2, learning_rate=0.2),
                judge=lambda entry: FTTTJudgeResult(
                    is_correct=False,
                    reward=0.0,
                    feedback_text="Your answer is incorrect. Recheck the arithmetic.",
                    feedback_token_ids=feedback_token_ids,
                ),
            )
            trace = GenerationTraceBatch(
                entries=(
                    CompletionTraceEntry(
                        request_id="req-fail",
                        prompt_idx=6,
                        sample_idx=0,
                        output_index=0,
                        token_ids=(8, 8),
                        text="5",
                        finish_reason="stop",
                        group_id="round-tokens",
                    ),
                )
            )

            consumer.run_toy_closed_loop_step(trace, torch.eye(1, 4))

            return consumer.engine.adapter.up.weight.detach().clone()

        update_from_first_tokens = run_once((11, 12, 13))
        update_from_second_tokens = run_once((31, 32, 33))

        self.assertFalse(torch.allclose(update_from_first_tokens, update_from_second_tokens))

    def test_toy_closed_loop_exports_request_scoped_lora_update_after_adapter_update(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
        from tllm.runtime.test_time_lora import TestTimeLoRAService

        class _FakeLoRAModule:
            def __init__(self) -> None:
                self.calls: list[tuple[int, torch.Tensor, torch.Tensor, float]] = []

            def set_lora(
                self,
                adapter_id: int,
                lora_a: torch.Tensor,
                lora_b: torch.Tensor,
                *,
                bias: torch.Tensor | None,
                scaling: float,
            ) -> None:
                self.calls.append((adapter_id, lora_a.detach().clone(), lora_b.detach().clone(), float(scaling)))

        class _FakeManager:
            def __init__(self, module: _FakeLoRAModule) -> None:
                self.modules = {"model.layers.2.mlp.down_proj": module}

        module = _FakeLoRAModule()
        consumer = FTTTConsumer(
            FTTTConsumerConfig(hidden_size=4, feedback_feature_size=4, rank=2, learning_rate=0.2, max_retries=2),
            judge=lambda entry: FTTTJudgeResult(
                is_correct=False,
                reward=0.0,
                feedback_text="Your answer is incorrect. Fix the sign.",
                feedback_token_ids=(44, 45),
            ),
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-lora",
                    prompt_idx=8,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(3, 5),
                    text="-4",
                    finish_reason="stop",
                    group_id="round-lora",
                ),
            )
        )

        result = consumer.run_toy_closed_loop_step(
            trace,
            torch.eye(1, 4),
            target_module_name="model.layers.2.mlp.down_proj",
            lora_int_id=29,
            adapter_key_prefix="fttt",
            alpha=2.0,
            step_id=12,
        )

        self.assertIsNotNone(result.lora_update)
        assert result.lora_update is not None
        self.assertEqual(result.lora_update.scope, "request")
        self.assertEqual(result.lora_update.request_ids, ("req-lora",))
        self.assertEqual(result.lora_update.adapter_key, "fttt:round-lora:8:v1")
        self.assertGreater(float(result.lora_update.target_modules["model.layers.2.mlp.down_proj"].lora_b.abs().sum()), 0.0)

        service = TestTimeLoRAService(manager=_FakeManager(module), max_rank=2)
        service.stage_update(result.lora_update)

        self.assertEqual(service.commit_staged_updates(), 1)
        self.assertEqual(service.adapter_for_request("req-lora"), 29)
        self.assertEqual(len(module.calls), 1)

    def test_failed_completion_builds_feedback_lm_training_batch(self) -> None:
        from tllm.consumers.fttt import FeedbackRecord, FTTTConsumer, FTTTConsumerConfig

        consumer = FTTTConsumer(FTTTConsumerConfig())
        batch = consumer.feedback_training_batch(
            (
                FeedbackRecord(
                    prompt_idx=8,
                    feedback="Your answer is incorrect. Fix the sign.",
                    reward=0.0,
                    request_id="req-train",
                    sample_idx=2,
                    output_index=1,
                    group_id="round-train",
                    failed_text="-4",
                    failed_token_ids=(31, 32),
                    feedback_token_ids=(41, 42, 43),
                ),
            )
        )

        self.assertEqual(batch.request_ids, ("req-train",))
        self.assertEqual(batch.metadata["consumer_id"], "fttt")
        example = batch.examples[0]
        self.assertEqual(example.request_id, "req-train")
        self.assertEqual(example.prompt_id, "8")
        self.assertEqual(example.sample_id, "2")
        self.assertEqual(example.output_id, "1")
        self.assertEqual(example.objective_kind, "feedback_lm")
        self.assertEqual(example.token_ids, (31, 32, 41, 42, 43))
        self.assertEqual(example.labels, (-100, -100, 41, 42, 43))
        self.assertEqual(example.rewards, (0.0,))
        self.assertEqual(example.metadata["failed_text"], "-4")
        self.assertEqual(example.metadata["failed_token_ids"], (31, 32))
        self.assertEqual(example.metadata["feedback_token_ids"], (41, 42, 43))
        self.assertEqual(example.metadata["group_id"], "round-train")
        self.assertEqual(example.metadata["feedback_text"], "Your answer is incorrect. Fix the sign.")

    def test_trainer_bridge_result_stages_lora_update_and_retry_references_adapter(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.contracts.test_time_training import TestTimeTrainerResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
        from tllm.runtime.test_time_lora import TestTimeLoRAService
        from tllm.runtime.test_time_trainer import FakeTestTimeTrainerBridge

        class _FakeLoRAModule:
            def __init__(self) -> None:
                self.calls: list[tuple[int, torch.Tensor, torch.Tensor, float]] = []

            def set_lora(
                self,
                adapter_id: int,
                lora_a: torch.Tensor,
                lora_b: torch.Tensor,
                *,
                bias: torch.Tensor | None,
                scaling: float,
            ) -> None:
                self.calls.append((adapter_id, lora_a.detach().clone(), lora_b.detach().clone(), float(scaling)))

        class _FakeManager:
            def __init__(self, module: _FakeLoRAModule) -> None:
                self.modules = {"model.layers.0.self_attn.q_proj": module}

        update = TestTimeLoRAUpdate(
            adapter_key="fttt:round-bridge:8:v7",
            lora_int_id=37,
            rank=1,
            scope="request",
            request_ids=("req-bridge",),
            target_modules={
                "model.layers.0.self_attn.q_proj": TestTimeLoRAWeights(
                    lora_a=torch.ones(3, 1),
                    lora_b=torch.ones(1, 4),
                    alpha=2.0,
                )
            },
            step_id=7,
        )
        module = _FakeLoRAModule()
        service = TestTimeLoRAService(manager=_FakeManager(module), max_rank=1)
        bridge = FakeTestTimeTrainerBridge(result=TestTimeTrainerResult(updates=(update,)), stage_updates=False)
        consumer = FTTTConsumer(
            FTTTConsumerConfig(max_retries=2),
            judge=lambda entry: FTTTJudgeResult(
                is_correct=False,
                reward=0.0,
                feedback_text="Your answer is incorrect. Fix the sign.",
                feedback_token_ids=(51, 52),
            ),
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-bridge",
                    prompt_idx=8,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(31, 32),
                    text="-4",
                    finish_reason="stop",
                    group_id="round-bridge",
                ),
            )
        )

        result = consumer.run_trainer_bridge_closed_loop_step(
            trace,
            trainer_bridge=bridge,
            lora_service=service,
            adapter_key_prefix="fttt",
            target_modules=("model.layers.0.self_attn.q_proj",),
            rank=1,
            step_id=7,
        )

        self.assertEqual(len(bridge.requests), 1)
        self.assertEqual(bridge.requests[0].examples.examples[0].metadata["feedback_token_ids"], (51, 52))
        self.assertEqual(result.staged_update_count, 1)
        self.assertEqual(service.commit_staged_updates(), 1)
        self.assertEqual(service.adapter_for_request("req-bridge"), 37)
        self.assertEqual(len(module.calls), 1)
        retry = result.retry_controls.entries[0]
        self.assertTrue(retry.retry_after_update)
        self.assertIn("adapter-key=fttt:round-bridge:8:v7", retry.reason)
        self.assertIn("adapter-version=7", retry.reason)

    def test_no_trainer_update_blocks_retry_after_update(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.contracts.test_time_training import TestTimeTrainerResult
        from tllm.runtime.generation_trace_runtime import CompletionTraceEntry, GenerationTraceBatch
        from tllm.runtime.test_time_trainer import FakeTestTimeTrainerBridge

        bridge = FakeTestTimeTrainerBridge(result=TestTimeTrainerResult(), stage_updates=False)
        consumer = FTTTConsumer(
            FTTTConsumerConfig(max_retries=2),
            judge=lambda entry: FTTTJudgeResult(
                is_correct=False,
                reward=0.0,
                feedback_text="Your answer is incorrect.",
                feedback_token_ids=(61,),
            ),
        )
        trace = GenerationTraceBatch(
            entries=(
                CompletionTraceEntry(
                    request_id="req-no-update",
                    prompt_idx=3,
                    sample_idx=0,
                    output_index=0,
                    token_ids=(9,),
                    text="bad",
                    finish_reason="stop",
                    group_id="round-empty",
                ),
            )
        )

        result = consumer.run_trainer_bridge_closed_loop_step(
            trace,
            trainer_bridge=bridge,
            adapter_key_prefix="fttt",
            target_modules=("model.layers.0.self_attn.q_proj",),
            rank=1,
            step_id=3,
        )

        self.assertEqual(result.staged_update_count, 0)
        retry = result.retry_controls.entries[0]
        self.assertEqual(retry.action, "retry")
        self.assertFalse(retry.retry_after_update)
        self.assertIn("trainer-no-update", retry.reason)

    def test_trace_judge_loop_rejects_non_strict_identity_fields(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FTTTJudgeResult
        from tllm.runtime.generation_trace_runtime import GenerationTraceBatch

        trace_entry = type(
            "LooseTraceEntry",
            (),
            {
                "request_id": 123,
                "prompt_idx": True,
                "sample_idx": 0,
                "output_index": 0,
                "group_id": "",
                "text": "bad",
            },
        )()
        consumer = FTTTConsumer(
            FTTTConsumerConfig(),
            judge=lambda entry: FTTTJudgeResult(is_correct=False, reward=0.0, feedback_text="incorrect"),
        )

        with self.assertRaisesRegex(TypeError, "request_id must be str"):
            consumer.process_generation_trace(GenerationTraceBatch(entries=(trace_entry,)))  # type: ignore[arg-type]

    def test_exports_feedback_adapter_as_strict_test_time_lora_update(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig
        from tllm.consumers.fttt import FeedbackRecord
        from tllm.contracts.test_time_lora import TestTimeLoRAUpdate, TestTimeLoRAWeights
        from tllm.runtime.test_time_lora import TestTimeLoRAService

        class _FakeLoRAModule:
            def __init__(self) -> None:
                self.calls: list[tuple[int, torch.Tensor, torch.Tensor, float]] = []

            def set_lora(
                self,
                adapter_id: int,
                lora_a: torch.Tensor,
                lora_b: torch.Tensor,
                *,
                bias: torch.Tensor | None,
                scaling: float,
            ) -> None:
                self.calls.append((adapter_id, lora_a.detach().clone(), lora_b.detach().clone(), float(scaling)))

        class _FakeManager:
            def __init__(self, module: _FakeLoRAModule) -> None:
                self.modules = {"model.layers.1.mlp.down_proj": module}

        torch.manual_seed(13)
        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, feedback_feature_size=3, rank=2, learning_rate=0.2))
        consumer.apply_feedback_update(
            torch.eye(2, 4),
            [
                FeedbackRecord(prompt_idx=0, feedback="incorrect arithmetic", reward=0.0),
                FeedbackRecord(prompt_idx=1, feedback="wrong final sign", reward=0.0),
            ],
        )

        update = consumer.export_test_time_lora_update(
            adapter_key="fttt-req-7",
            lora_int_id=23,
            target_module_name="model.layers.1.mlp.down_proj",
            rank=2,
            scope="request",
            request_ids=("req-7", "req-8"),
            step_id=9,
            alpha=2.5,
        )

        self.assertIsInstance(update, TestTimeLoRAUpdate)
        self.assertEqual(update.adapter_key, "fttt-req-7")
        self.assertEqual(update.lora_int_id, 23)
        self.assertEqual(update.rank, 2)
        self.assertEqual(update.scope, "request")
        self.assertEqual(update.request_ids, ("req-7", "req-8"))
        self.assertEqual(update.step_id, 9)
        self.assertEqual(tuple(update.target_modules), ("model.layers.1.mlp.down_proj",))
        weights = update.target_modules["model.layers.1.mlp.down_proj"]
        self.assertIsInstance(weights, TestTimeLoRAWeights)
        self.assertEqual(tuple(weights.lora_a.shape), (8, 2))
        self.assertEqual(tuple(weights.lora_b.shape), (2, 4))
        self.assertEqual(weights.alpha, 2.5)
        self.assertGreater(float(weights.lora_b.abs().sum()), 0.0)

        module = _FakeLoRAModule()
        service = TestTimeLoRAService(manager=_FakeManager(module), max_rank=2)
        service.stage_update(update)

        self.assertEqual(service.commit_staged_updates(), 1)
        self.assertEqual(service.adapter_for_request("req-7"), 23)
        self.assertEqual(service.adapter_for_request("req-8"), 23)
        self.assertEqual(len(module.calls), 1)
        adapter_id, committed_lora_a, committed_lora_b, scaling = module.calls[0]
        self.assertEqual(adapter_id, 23)
        self.assertTrue(torch.equal(committed_lora_a, weights.lora_a))
        self.assertTrue(torch.equal(committed_lora_b, weights.lora_b))
        self.assertEqual(scaling, 1.25)

    def test_feedback_lora_export_rejects_rank_target_and_request_boundary_mistakes(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig

        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, feedback_feature_size=3, rank=2))

        with self.assertRaisesRegex(ValueError, "adapter rank"):
            consumer.export_test_time_lora_update(
                adapter_key="fttt-rank",
                lora_int_id=1,
                target_module_name="model.layers.1.mlp.down_proj",
                rank=4,
            )

        with self.assertRaisesRegex(TypeError, "request_ids must be tuple"):
            consumer.export_test_time_lora_update(
                adapter_key="fttt-ids",
                lora_int_id=2,
                target_module_name="model.layers.1.mlp.down_proj",
                rank=2,
                scope="request",
                request_ids=["req-7"],  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(TypeError, "request_ids must contain str"):
            consumer.export_test_time_lora_update(
                adapter_key="fttt-id-member",
                lora_int_id=2,
                target_module_name="model.layers.1.mlp.down_proj",
                rank=2,
                scope="request",
                request_ids=(7,),  # type: ignore[arg-type]
            )

        with self.assertRaisesRegex(TypeError, "lora_int_id must be int"):
            consumer.export_test_time_lora_update(
                adapter_key="fttt-bool-id",
                lora_int_id=True,  # type: ignore[arg-type]
                target_module_name="model.layers.1.mlp.down_proj",
                rank=2,
            )

        with self.assertRaisesRegex(ValueError, "target module name"):
            consumer.export_test_time_lora_update(
                adapter_key="fttt-target",
                lora_int_id=4,
                target_module_name="",
                rank=2,
            )

    def test_fttt_module_does_not_import_external_feedback_apis(self) -> None:
        before = set(sys.modules)

        __import__("tllm.consumers.fttt")

        imported = set(sys.modules) - before
        self.assertNotIn("requests", imported)
        self.assertNotIn("fastapi", imported)


if __name__ == "__main__":
    unittest.main()
