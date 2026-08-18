#!/usr/bin/env python3
"""Unit tests for the In-Place-TTT-inspired consumer."""

from __future__ import annotations

import unittest

import torch

from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.runtime_context import RuntimeContext


def _ctx() -> RuntimeContext:
    return RuntimeContext(
        runner=None,
        model=None,
        device=torch.device("cpu"),
        main_stream=None,
        is_compiling=False,
        uses_cudagraph=False,
        event_name="unit",
    )


class InPlaceTTTConsumerUnitTest(unittest.TestCase):
    def test_enabled_consumer_defaults_to_prefill_hidden_pair_flow(self) -> None:
        from tllm.consumers.inplace_ttt import InPlaceTTTConsumer, InPlaceTTTConsumerConfig

        consumer = InPlaceTTTConsumer(InPlaceTTTConsumerConfig(enabled=True))

        flows = tuple(consumer.flows())
        self.assertEqual(len(flows), 1)
        self.assertEqual(flows[0].delivery, "bundle")
        self.assertEqual(flows[0].ownership, "borrowed")
        self.assertEqual(flows[0].bundle_key, ("engine_step_id", "phase"))
        phases = [read.locator.phase for read in flows[0].reads if getattr(read.locator, "phase", None)]
        self.assertEqual(phases, ["prefill", "prefill"])

    def test_consume_bundle_waits_for_full_chunk_and_carries_tail_between_calls(self) -> None:
        from tllm.consumers.inplace_ttt import InPlaceTTTConsumer, InPlaceTTTConsumerConfig

        consumer = InPlaceTTTConsumer(
            InPlaceTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1, chunk_size=4)
        )
        source = torch.randn(3, 4)
        target = source + 0.25
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="prefill", request_id="", sample_idx=0),
            entries={"source": source, "target": target},
        )

        consumer.consume_bundle(bundle, _ctx())

        stats = consumer.read_stats()
        self.assertEqual(stats.update_count, 0)
        self.assertEqual(stats.loss_count, 0)
        self.assertEqual(stats.trained_rows, 0)
        self.assertEqual(stats.tail_rows, 3)

        next_source = torch.randn(1, 4)
        next_target = next_source + 0.25
        next_bundle = PortBundle(
            key=BundleKey(engine_step_id=2, phase="prefill", request_id="", sample_idx=0),
            entries={"source": next_source, "target": next_target},
        )

        consumer.consume_bundle(next_bundle, _ctx())

        stats = consumer.read_stats()
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.loss_count, 1)
        self.assertEqual(stats.trained_rows, 4)
        self.assertEqual(stats.tail_rows, 0)


if __name__ == "__main__":
    unittest.main()
