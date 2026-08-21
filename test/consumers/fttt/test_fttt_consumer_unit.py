#!/usr/bin/env python3
"""Unit tests for the FTTT-inspired consumer."""

from __future__ import annotations

import unittest

import torch


class FTTTConsumerUnitTest(unittest.TestCase):
    def test_package_exports_fttt_consumer(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig

        consumer = FTTTConsumer(FTTTConsumerConfig())

        self.assertEqual(consumer.consumer_id, "fttt")

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

    def test_training_actually_updates_bank_params(self) -> None:
        from tllm.consumers.fttt import FTTTConsumer, FTTTConsumerConfig, FeedbackRecord

        torch.manual_seed(0)
        consumer = FTTTConsumer(FTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.2))
        hidden = torch.randn(2, 4)
        feedback = [FeedbackRecord(prompt_idx=0, feedback="wrong arithmetic carry", reward=0.0)] * 2
        consumer.apply_feedback_update(hidden, feedback)
        compiled = consumer.compiled
        assert compiled is not None
        slot = compiled.bank("edit_adapter").slot_for(None)
        up = slot.params["adapter.up.weight"]
        self.assertGreater(float(up.abs().max()), 0.0)
        self.assertEqual(consumer.read_stats().update_count, 1)

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


if __name__ == "__main__":
    unittest.main()
