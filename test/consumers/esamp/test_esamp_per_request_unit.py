#!/usr/bin/env python3
"""Unit tests for per-request ESamp helpers."""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from tllm.consumers.esamp import engine as engine_module
from tllm.consumers.esamp.config import AdaptationStreamMode, ESampConsumerConfig
from tllm.consumers.esamp.engine import ESampTrainEngine, group_row_indices_by_prompt
from tllm.consumers.esamp.model import LowRankGatedResidualModel
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

    def test_legacy_per_request_prompt_key_oracle(self) -> None:
        """Freeze the non-bank legacy key, update, and retention semantics.

        The legacy engine receives only ``prompt_idxs``.  Consequently all
        parallel sample rows for one prompt share a trainer and produce one
        loss/optimizer update; a different prompt owns separate model and
        optimizer state.  There is no request-finished input in this engine:
        reading per-prompt metrics resets counters but retains those owners.
        """
        torch.manual_seed(17)
        reference_a = LowRankGatedResidualModel(
            hidden_size=4,
            rank=2,
            use_output_layernorm=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).train()
        LowRankGatedResidualModel(
            hidden_size=4,
            rank=2,
            use_output_layernorm=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).eval()
        reference_b = LowRankGatedResidualModel(
            hidden_size=4,
            rank=2,
            use_output_layernorm=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).train()
        LowRankGatedResidualModel(
            hidden_size=4,
            rank=2,
            use_output_layernorm=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ).eval()

        torch.manual_seed(17)
        consumer = ESampTrainEngine(
            hidden_dim=2,
            lr=1e-2,
            enabled=True,
            per_request_models=True,
            per_request_model_bank=False,
        )
        with mock.patch("torch.cuda.Stream"), mock.patch("torch.cuda.Event"), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ), mock.patch("torch.cuda.current_stream", return_value=mock.Mock()):
            consumer.ensure_resources(
                device=torch.device("cpu"),
                rows=4,
                hidden_size=4,
                hidden_dtype=torch.float32,
            )

        prompt_a = 71
        prompt_b = 113
        source_a = torch.tensor(
            [
                [0.1, -0.2, 0.3, -0.4],
                [0.5, 0.6, -0.7, 0.8],
                [-0.9, 1.0, 1.1, -1.2],
            ],
            dtype=torch.float32,
        )
        target_a = torch.tensor(
            [
                [-0.3, 0.2, 0.1, 0.4],
                [0.8, -0.6, 0.5, -0.2],
                [0.7, 0.9, -1.0, 1.1],
            ],
            dtype=torch.float32,
        )
        expected_a_loss = F.mse_loss(reference_a(source_a).float(), target_a.float())

        with mock.patch("torch.cuda.current_stream", return_value=mock.Mock()), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.launch_step(source_a, target_a)
            consumer.launch_delayed_backward(3, (prompt_a, prompt_a, prompt_a))

        self.assertEqual(set(consumer.state.per_request_trainers), {prompt_a})
        trainer_a = consumer.state.per_request_trainers[prompt_a]
        self.assertFalse(torch.equal(trainer_a.train_model.a, reference_a.a))
        self.assertTrue(torch.equal(trainer_a.forward_model.a, trainer_a.train_model.a))

        a_after_prompt_a = trainer_a.train_model.a.detach().clone()
        source_b = torch.tensor(
            [[1.0, -0.5, 0.25, -0.75], [-0.1, 0.2, -0.3, 0.4]],
            dtype=torch.float32,
        )
        target_b = torch.tensor(
            [[-0.4, 0.3, -0.2, 0.1], [0.9, -0.8, 0.7, -0.6]],
            dtype=torch.float32,
        )
        expected_b_loss = F.mse_loss(reference_b(source_b).float(), target_b.float())

        with mock.patch("torch.cuda.current_stream", return_value=mock.Mock()), mock.patch(
            "torch.cuda.stream", return_value=contextlib.nullcontext()
        ):
            consumer.launch_step(source_b, target_b)
            consumer.launch_delayed_backward(2, (prompt_b, prompt_b))

        trainer_b = consumer.state.per_request_trainers[prompt_b]
        self.assertIsNot(trainer_a.train_model, trainer_b.train_model)
        self.assertIsNot(trainer_a.optimizer, trainer_b.optimizer)
        self.assertTrue(torch.equal(trainer_a.train_model.a, a_after_prompt_a))
        self.assertFalse(torch.equal(trainer_b.train_model.a, reference_b.a))
        self.assertTrue(torch.equal(trainer_b.forward_model.a, trainer_b.train_model.a))
        stats = consumer.read_and_reset_stats(sync=False)
        self.assertEqual(stats.loss_count, 2)
        self.assertAlmostEqual(
            stats.loss_avg,
            float(((expected_a_loss + expected_b_loss) / 2).item()),
            places=6,
        )

        per_request = consumer.read_and_reset_per_request_stats(sync=False)
        self.assertEqual(set(per_request), {prompt_a, prompt_b})
        self.assertEqual(per_request[prompt_a].loss_count, 1)
        self.assertEqual(per_request[prompt_b].loss_count, 1)
        # Metric reset is not request cleanup: both prompt-keyed trainers persist.
        self.assertIs(consumer.state.per_request_trainers[prompt_a], trainer_a)
        self.assertIs(consumer.state.per_request_trainers[prompt_b], trainer_b)
        retained = consumer.read_and_reset_per_request_stats(sync=False)
        self.assertEqual(set(retained), {prompt_a, prompt_b})
        self.assertEqual(retained[prompt_a].loss_count, 0)
        self.assertEqual(retained[prompt_b].loss_count, 0)

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
