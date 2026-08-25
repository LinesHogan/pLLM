#!/usr/bin/env python3
"""Unit tests for distiller precompute scheduling."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.backends.vllm.sampler_mrv2 import (
    _build_mrv2_step_view,
    _record_mrv2_candidate_stats,
)
from tllm.runtime.residual_runtime import SamplerPrecomputeState
from tllm.runtime.sampler_bridge.runtime_view import build_sampler_step_view
from tllm.runtime.vllm_patch import sampler_patch


class SamplerPrecomputeUnitTest(unittest.TestCase):
    def test_mrv1_step_view_keeps_host_identity_with_device_index_tensors(self) -> None:
        runtime = self._runtime(
            decode_count=2,
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.ones((2, 4))},
            decode_prompt_idx_tensor=torch.tensor([7, 8]),
            decode_sample_idx_tensor=torch.tensor([0, 1]),
            decode_prompt_idxs=[7, 8],
            decode_sample_idxs=[0, 1],
            decode_request_ids=["req-a", "req-b"],
            event_step_id=11,
        )
        runner = SimpleNamespace(model=object())

        view = build_sampler_step_view(
            runtime=runtime,
            runner=runner,
            model=runner.model,
            logits=torch.ones((2, 16)),
            sampling_metadata=object(),
        )

        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(view.request_ids, ("req-a", "req-b"))
        self.assertEqual(view.prompt_idxs, (7, 8))
        self.assertEqual(view.sample_idxs, (0, 1))
        self.assertIsNotNone(view.prompt_idx_tensor)
        self.assertIsNotNone(view.sample_idx_tensor)

    def test_mrv2_step_view_keeps_host_identity_with_device_index_tensors(self) -> None:
        runtime = self._runtime(
            decode_count=2,
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.ones((2, 4))},
            decode_prompt_idx_tensor=torch.tensor([7, 8]),
            decode_sample_idx_tensor=torch.tensor([0, 1]),
            decode_prompt_idxs=[7, 8],
            decode_sample_idxs=[0, 1],
            decode_request_ids=["req-a", "req-b"],
            event_step_id=11,
        )
        runner = SimpleNamespace(model=object())

        view = _build_mrv2_step_view(
            runtime=runtime,
            runner=runner,
            logits=torch.ones((2, 16)),
        )

        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(view.request_ids, ("req-a", "req-b"))
        self.assertEqual(view.prompt_idxs, (7, 8))
        self.assertEqual(view.sample_idxs, (0, 1))
        self.assertIsNotNone(view.prompt_idx_tensor)
        self.assertIsNotNone(view.sample_idx_tensor)

    def tearDown(self) -> None:
        sampler_patch._ORIG_VLLM_SAMPLER_SAMPLE = None

    def _runtime(self, **kwargs):
        kwargs.setdefault("sampler_precompute", SamplerPrecomputeState())
        return SimpleNamespace(**kwargs)

    def test_mrv2_candidate_fast_path_records_without_device_readback(self) -> None:
        runtime = self._runtime()

        _record_mrv2_candidate_stats(runtime, candidate_count=64, row_count=8)

        stats = runtime.sampler_precompute
        self.assertEqual(stats.candidate_sample_count, 1)
        self.assertEqual(stats.candidate_token_count, 64)
        self.assertEqual(stats.candidate_row_count, 8)
        self.assertEqual(stats.candidate_max_count, 64)
        self.assertEqual(stats.candidate_kernel_triton_count, 1)

    def test_maybe_schedule_sampler_precompute_populates_runtime_cache(self) -> None:
        engine = mock.Mock()
        engine.predict_hidden_for_sampling.return_value = (
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        )
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=11,
            decode_count=2,
            decode_prompt_idxs=[7, 8],
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)},
            sampler_precompute=SamplerPrecomputeState(stream=None, event=None, precomputed_step_id=-1),
        )
        runner = SimpleNamespace(
            _tllm_runtime=runtime,
            model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)),
        )

        sampler_patch.maybe_schedule_sampler_precompute(runtime=runtime, runner=runner, layer_path="layers.0")

        engine.predict_hidden_for_sampling.assert_called_once()
        self.assertEqual(runtime.sampler_precompute.precomputed_step_id, 11)
        cache = runtime.sampler_precompute.cache_for_step(11)
        assert cache is not None
        self.assertTrue(torch.equal(cache.row_ids, torch.tensor([0], dtype=torch.long)))
        self.assertTrue(torch.equal(cache.pred_hidden, torch.tensor([[5.0, 6.0]], dtype=torch.float32)))
        self.assertIsNone(cache.dense_logits)

    def test_capture_source_precompute_owns_buffer_lifecycle(self) -> None:
        engine = mock.Mock()

        def _capture(source_hidden, prompt_idxs, *, out_pred_hidden, out_valid_mask):
            out_pred_hidden.copy_(source_hidden + 1.0)
            out_valid_mask.fill_(True)
            return True

        engine.predict_hidden_for_sampling_capture.side_effect = _capture
        consumer = ESampConsumer(
            ESampConsumerConfig(
                enable_esamp_training=True,
                enable_distiller_intervention=True,
                distiller_beta=0.5,
            ),
            engine=engine,
        )
        source_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        runtime = self._runtime(
            consumer=consumer,
            event_step_id=21,
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": source_hidden},
            decode_prompt_idx_buf=torch.tensor([7, 8], dtype=torch.long),
            sampler_allow_source_capture=True,
            sampler_allow_source_async=False,
            sampler_precompute=SamplerPrecomputeState(stream=None, event=None),
        )
        runner = SimpleNamespace(
            _tllm_runtime=runtime,
            model=SimpleNamespace(lm_head=SimpleNamespace(weight=torch.ones((3, 2)), bias=None)),
        )

        consumer.sampler_modifier_provider().maybe_capture_source_precompute(
            runtime=runtime,
            runner=runner,
            layer_path="layers.0",
        )

        buffers = runtime.sampler_precompute.buffers
        assert buffers is not None
        self.assertTrue(torch.equal(buffers.source_hidden, source_hidden))
        self.assertTrue(torch.equal(buffers.prompt_idx, torch.tensor([7, 8], dtype=torch.long)))
        self.assertTrue(torch.equal(buffers.pred_hidden, source_hidden + 1.0))
        self.assertTrue(torch.equal(buffers.valid_mask, torch.tensor([True, True])))
        self.assertEqual(runtime.sampler_precompute.source_capture_step_id, 21)


if __name__ == "__main__":
    unittest.main()
