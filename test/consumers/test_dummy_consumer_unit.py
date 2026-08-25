#!/usr/bin/env python3
"""Unit tests for the dummy consumer extension template."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.runtime_context import RuntimeContext
from tllm.consumers.dummy.config import DummyConsumerConfig
from tllm.consumers.dummy.consumer import DummyConsumer
from tllm.ports.base import PortKind


class DummyConsumerUnitTest(unittest.TestCase):
    def test_dummy_declares_background_flow_over_residual_and_request_meta(self) -> None:
        c = DummyConsumer(DummyConsumerConfig())
        flows = c.flows()

        self.assertEqual(len(flows), 1)
        self.assertFalse(hasattr(c, "consume"))
        self.assertFalse(hasattr(c, "on_tick"))
        flow = flows[0]
        self.assertEqual(flow.window, "background")
        self.assertEqual([read.kind for read in flow.reads], [PortKind.RESIDUAL_STREAM, PortKind.REQUEST_META])
        self.assertEqual([write.kind for write in flow.writes], [PortKind.CPU_EXPORT])
        self.assertEqual(flow.bundle_key, ("engine_step_id", "phase"))

    def test_dummy_default_path_stages_cpu_without_hot_path_drain(self) -> None:
        c = DummyConsumer(DummyConsumerConfig(export_every_n_steps=1))
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "hidden": torch.randn((2, 4), dtype=torch.float32),
                "request_meta": {"request_id": "reqA", "prompt_idx": 0, "sample_idx": 0},
            },
        )
        ctx = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:background",
        )

        with mock.patch("tllm.consumers.dummy.consumer.clone_hidden_to_cpu", wraps=lambda x: x.detach().to("cpu")) as p_clone, mock.patch(
            "builtins.print"
        ) as p_print:
            c.consume_bundle(bundle, ctx)
            hot_path_stats = c.read_stats()

        self.assertTrue(p_clone.called)
        self.assertFalse(p_print.called)
        self.assertEqual(hot_path_stats["consumed_batches"], 1.0)
        self.assertEqual(hot_path_stats["consumed_rows"], 2.0)
        self.assertEqual(hot_path_stats["processed_batches"], 0.0)
        self.assertEqual(hot_path_stats["pending"], 1.0)

        with mock.patch("builtins.print") as p_print:
            c.synchronize()

        stats = c.read_stats()
        self.assertTrue(p_print.called)
        self.assertEqual(stats["processed_batches"], 1.0)
        self.assertEqual(stats["pending"], 0.0)

    def test_legacy_feedback_interval_drains_async_exports_at_step_end(self) -> None:
        consumer = DummyConsumer(
            DummyConsumerConfig(
                enable_async=True,
                export_every_n_steps=1,
                feedback_interval=1,
            )
        )
        context = RuntimeContext(
            runner=None,
            model=None,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="step.end",
        )

        consumer.execute_native_export(torch.randn(2, 4), step_id=1)
        self.assertEqual(consumer.read_stats()["pending"], 1.0)
        consumer.on_step_end(context)

        self.assertEqual(consumer.read_stats()["processed_batches"], 1.0)
        self.assertEqual(consumer.read_stats()["pending"], 0.0)


if __name__ == "__main__":
    unittest.main()
