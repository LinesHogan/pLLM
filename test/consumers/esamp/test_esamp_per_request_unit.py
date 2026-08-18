#!/usr/bin/env python3
"""Unit tests for per-request ESamp helpers."""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

import torch

from tllm.consumers.esamp import engine as engine_module
from tllm.consumers.esamp.config import AdaptationStreamMode, ESampConsumerConfig
from tllm.consumers.esamp.engine import ESampTrainEngine, group_row_indices_by_prompt
from tllm.runtime import residual_runtime as esamp_runtime


def _graph_buffers(
    *,
    device: torch.device,
    rows: int,
    hidden: int,
    slots: int | None = None,
    valid: torch.Tensor | None = None,
    loss_value: float = 0.0,
    src: torch.Tensor | None = None,
    tgt: torch.Tensor | None = None,
) -> engine_module._GraphBuffers:
    slot_buffers = None
    if slots is not None:
        slot_buffers = engine_module._GraphSlotBuffers(
            slot_ids=torch.zeros((rows,), device=device, dtype=torch.long),
            slot_sum=torch.zeros((slots,), device=device, dtype=torch.float32),
            slot_cnt=torch.zeros((slots,), device=device, dtype=torch.float32),
        )
    return engine_module._GraphBuffers(
        pair=engine_module.GraphCopyPair(
            src=src if src is not None else torch.zeros((rows, hidden), device=device, dtype=torch.float32),
            tgt=tgt if tgt is not None else torch.zeros((rows, hidden), device=device, dtype=torch.float32),
        ),
        valid=valid if valid is not None else torch.zeros((rows,), device=device, dtype=torch.float32),
        loss=torch.full((1,), loss_value, device=device, dtype=torch.float32),
        slots=slot_buffers,
    )


class ESampPerRequestUnitTest(unittest.TestCase):
    class _ReplayFailGraph:
        def replay(self) -> None:
            raise RuntimeError("inference tensor update outside InferenceMode")

    def test_group_rows_by_prompt(self) -> None:
        prompt_idxs = [2, 2, -1, 5, 2, 5]
        grouped = group_row_indices_by_prompt(prompt_idxs, active_rows=6)
        self.assertEqual(grouped, {2: [0, 1, 4], 5: [3, 5]})
        self.assertEqual(group_row_indices_by_prompt([0, 1, 2, 3], active_rows=2), {0: [0], 1: [1]})

    def test_configure_runtime_keeps_only_runtime_owned_fields(self) -> None:
        esamp_runtime.configure_runtime(
            graph_scratch_rows=64,
            tap_layer_paths=["model.model.layers[0]", "model.model.layers[-1]"],
            source_layer_path="model.model.layers[0]",
            target_layer_path="model.model.layers[-1]",
            adaptation_pipeline_slots=8,
            adaptation_stream_mode="single",
            adaptation_stream_priority=-1,
        )
        cfg = esamp_runtime.RUNTIME.config
        self.assertEqual(cfg.adaptation_pipeline_slots, 8)
        self.assertEqual(cfg.adaptation_stream_mode, "single")
        self.assertEqual(cfg.adaptation_stream_priority, -1)
        # Consumer-algorithm fields no longer live on the runtime config.
        for removed in (
            "enable_esamp_training",
            "per_request_models",
            "per_request_model_bank",
            "model_bank_slots",
            "distiller_beta",
        ):
            self.assertFalse(hasattr(cfg, removed), removed)

    def test_adaptation_stream_mode_accepts_only_canonical_names(self) -> None:
        for mode in ("dual", "single", "serial"):
            with self.subTest(mode=mode):
                cfg = ESampConsumerConfig(adaptation_stream_mode=mode)
                self.assertEqual(cfg.adaptation_stream_mode, mode)

        cfg = ESampConsumerConfig(adaptation_stream_mode=AdaptationStreamMode.SINGLE)
        self.assertEqual(cfg.adaptation_stream_mode, AdaptationStreamMode.SINGLE)
        self.assertEqual(str(cfg.adaptation_stream_mode), "single")

        for mode in ("async", "async-dual", "dual-stream", "current", "main", "single-stream"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "adaptation_stream_mode"):
                    ESampConsumerConfig(adaptation_stream_mode=mode)

    def test_engine_model_bank_training_accumulates_nonzero_slot_loss_promoted_core(self) -> None:
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-3,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=True,
            model_bank_slots=2,
            model_bank_rank=2,
            model_bank_train_cudagraph=False,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=6,
                hidden_dtype=torch.float32,
            )

        src = torch.randn((4, 6), dtype=torch.float32)
        tgt = src + 0.25
        consumer._train_model_bank(src, tgt, 4, [0, 0, 1, 1])

        stats = consumer._require_stats()
        self.assertGreater(float(stats.loss_sum.item()), 0.0)
        self.assertEqual(int(stats.loss_count.item()), 2)

    def test_shared_graph_replay_failure_disables_graph_and_falls_back(self) -> None:
        device = torch.device("cpu")
        dtype = torch.float32
        hidden = 4
        rows = 3
        consumer = ESampTrainEngine(hidden_dim=2, lr=1e-3, enabled=True)
        shared_graph = consumer.state.graphs["shared"]
        shared_graph.capture_state = "captured"
        shared_graph.replay_graph = self._ReplayFailGraph()
        shared_graph.rows = rows
        shared_graph.buffers = _graph_buffers(device=device, rows=rows, hidden=hidden)
        consumer.state.stats = engine_module._StatsBuffers(
            loss_sum=torch.zeros((1,), device=device, dtype=torch.float32),
            loss_count=torch.zeros((1,), device=device, dtype=torch.int64),
        )
        src = torch.randn((rows, hidden), device=device, dtype=dtype)
        tgt = torch.randn((rows, hidden), device=device, dtype=dtype)

        ok = consumer._train_shared_batch_graph_locked(src, tgt, active_rows=rows)
        self.assertFalse(ok)
        self.assertEqual(consumer.state.graphs["shared"].capture_state, "disabled")
        self.assertIn("InferenceMode", consumer.state.graphs["shared"].disable_reason)

    def test_launch_forward_rejects_second_step_before_previous_step_is_flushed(self) -> None:
        consumer = ESampTrainEngine(hidden_dim=4, lr=1e-3, enabled=True)
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=8,
                hidden_dtype=torch.float32,
            )

        rows = torch.zeros((1, 8), device=torch.device("cpu"), dtype=torch.float32)
        with mock.patch("torch.cuda.current_stream", return_value=mock.Mock()), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.launch_forward(rows)
            with self.assertRaisesRegex(RuntimeError, "launch order"):
                consumer.launch_forward(rows)


if __name__ == "__main__":
    unittest.main()
