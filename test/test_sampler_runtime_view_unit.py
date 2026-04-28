#!/usr/bin/env python3
"""Unit tests for sampler runtime view assembly."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from tllm.runtime.sampler_bridge.runtime_view import build_sampler_step_view


class SamplerRuntimeViewUnitTest(unittest.TestCase):
    def test_build_sampler_step_view_preserves_request_ids_with_device_prompt_sample_tensors(self) -> None:
        logits = torch.randn(2, 4)
        runtime = SimpleNamespace(
            event_step_id=12,
            decode_count=2,
            decode_request_ids=["0_parent", "1_parent"],
            decode_prompt_idxs=[],
            decode_sample_idxs=[],
            decode_prompt_idx_tensor=torch.tensor([3, 4], dtype=torch.long),
            decode_sample_idx_tensor=torch.tensor([0, 1], dtype=torch.long),
            source_resolved_path="layers.0",
            tap_decode_hidden={"layers.0": torch.randn(2, 3)},
        )

        view = build_sampler_step_view(
            runtime=runtime,
            runner=SimpleNamespace(),
            model=SimpleNamespace(),
            logits=logits,
            sampling_metadata=SimpleNamespace(),
        )

        self.assertIsNotNone(view)
        assert view is not None
        self.assertEqual(view.request_ids, ("0_parent", "1_parent"))
        self.assertIsNotNone(view.prompt_idx_tensor)
        self.assertIsNotNone(view.sample_idx_tensor)


if __name__ == "__main__":
    unittest.main()
