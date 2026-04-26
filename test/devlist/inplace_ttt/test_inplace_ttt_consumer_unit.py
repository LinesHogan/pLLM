#!/usr/bin/env python3
"""Unit tests for the In-Place-TTT-inspired devlist consumer."""

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
    def test_package_exports_inplace_ttt_consumer(self) -> None:
        from tllm.consumers.devlist import InPlaceTTTConsumer, InPlaceTTTConsumerConfig

        consumer = InPlaceTTTConsumer(InPlaceTTTConsumerConfig())

        self.assertEqual(consumer.consumer_id, "inplace_ttt")

    def test_enabled_consumer_defaults_to_prefill_hidden_pair_flow(self) -> None:
        from tllm.consumers.devlist import InPlaceTTTConsumer, InPlaceTTTConsumerConfig

        consumer = InPlaceTTTConsumer(InPlaceTTTConsumerConfig(enabled=True))

        flows = tuple(consumer.flows())
        self.assertEqual(len(flows), 1)
        self.assertEqual(flows[0].delivery, "bundle")
        self.assertEqual(flows[0].ownership, "borrowed")
        self.assertEqual(flows[0].bundle_key, ("engine_step_id", "phase"))
        phases = [read.locator.phase for read in flows[0].reads if getattr(read.locator, "phase", None)]
        self.assertEqual(phases, ["prefill", "prefill"])

    def test_decode_flow_is_explicit_optional_path(self) -> None:
        from tllm.consumers.devlist import InPlaceTTTConsumer, InPlaceTTTConsumerConfig

        consumer = InPlaceTTTConsumer(InPlaceTTTConsumerConfig(enabled=True, phases=("prefill", "decode")))

        flows = tuple(consumer.flows())
        self.assertEqual(len(flows), 2)
        self.assertEqual(flows[0].delivery, "bundle")
        self.assertEqual(flows[1].delivery, "device_lease")
        decode_phases = [read.locator.phase for read in flows[1].reads if getattr(read.locator, "phase", None)]
        self.assertEqual(decode_phases, ["decode", "decode"])

    def test_pad_to_chunks_extends_rows_to_chunk_multiple(self) -> None:
        from tllm.consumers.devlist.inplace_ttt import pad_rows_to_chunk

        rows = torch.arange(10, dtype=torch.float32).reshape(5, 2)

        padded, valid_rows = pad_rows_to_chunk(rows, chunk_size=4)

        self.assertEqual(valid_rows, 5)
        self.assertEqual(tuple(padded.shape), (8, 2))
        self.assertTrue(torch.equal(padded[:5], rows))
        self.assertTrue(torch.equal(padded[5:], torch.zeros_like(padded[5:])))

    def test_consume_bundle_updates_side_adapter_stats(self) -> None:
        from tllm.consumers.devlist import InPlaceTTTConsumer, InPlaceTTTConsumerConfig

        consumer = InPlaceTTTConsumer(InPlaceTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        source = torch.randn(3, 4)
        target = source + 0.25
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="prefill", request_id="", sample_idx=0),
            entries={"source": source, "target": target},
        )

        consumer.consume_bundle(bundle, _ctx())

        stats = consumer.read_stats()
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.loss_count, 1)
        self.assertEqual(stats.trained_rows, 3)

    def test_update_accepts_inference_mode_rows(self) -> None:
        from tllm.consumers.devlist import InPlaceTTTConsumer, InPlaceTTTConsumerConfig

        consumer = InPlaceTTTConsumer(InPlaceTTTConsumerConfig(hidden_size=4, rank=2, learning_rate=0.1))
        with torch.inference_mode():
            source = torch.randn(2, 4)
            target = source + 0.5
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="prefill", request_id="", sample_idx=0),
            entries={"source": source, "target": target},
        )

        consumer.consume_bundle(bundle, _ctx())

        self.assertEqual(consumer.read_stats().update_count, 1)

    def test_disabled_consumer_does_not_update(self) -> None:
        from tllm.consumers.devlist import InPlaceTTTConsumer, InPlaceTTTConsumerConfig

        consumer = InPlaceTTTConsumer(InPlaceTTTConsumerConfig(enabled=False, hidden_size=4))
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="prefill", request_id="", sample_idx=0),
            entries={"source": torch.randn(2, 4), "target": torch.randn(2, 4)},
        )

        consumer.consume_bundle(bundle, _ctx())

        self.assertEqual(consumer.read_stats().update_count, 0)


if __name__ == "__main__":
    unittest.main()
