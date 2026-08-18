#!/usr/bin/env python3
"""Unit tests for min-p candidate filtering helpers."""

from __future__ import annotations

import unittest

import pytest
import torch

from tllm.runtime.sampler_bridge.minp_kernel import TritonMinPCandidateKernel
from tllm.runtime.sampler_bridge.min_p import apply_min_p, min_p_keep_mask


class SamplerBridgeMinPUnitTest(unittest.TestCase):
    def test_apply_min_p_keeps_tokens_above_scaled_max_probability(self) -> None:
        logits = torch.log(torch.tensor([[0.60, 0.30, 0.06, 0.04]], dtype=torch.float32))
        min_p = torch.tensor([0.2], dtype=torch.float32)

        filtered = apply_min_p(logits.clone(), min_p)

        self.assertTrue(torch.isfinite(filtered[0, 0]))
        self.assertTrue(torch.isfinite(filtered[0, 1]))
        self.assertFalse(torch.isfinite(filtered[0, 2]))
        self.assertFalse(torch.isfinite(filtered[0, 3]))

    def test_min_p_keep_mask_matches_apply_min_p_finite_positions(self) -> None:
        logits = torch.tensor([[2.0, 1.8, -5.0, -6.0], [0.1, 3.0, 2.9, -7.0]], dtype=torch.float32)
        min_p = torch.tensor([0.05, 0.05], dtype=torch.float32)

        mask = min_p_keep_mask(logits, min_p)
        filtered = apply_min_p(logits.clone(), min_p)

        self.assertTrue(torch.equal(mask, torch.isfinite(filtered)))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton min-p kernel tests")
    def test_triton_minp_keep_mask_matches_torch_reference_on_cuda(self) -> None:
        logits = torch.tensor(
            [[2.0, 1.8, -5.0, -6.0], [0.1, 3.0, 2.9, -7.0]],
            dtype=torch.float32,
            device="cuda",
        )
        min_p = torch.tensor([0.05, 0.05], dtype=torch.float32, device="cuda")

        kernel = TritonMinPCandidateKernel()
        mask = kernel.keep_mask(logits, min_p)

        self.assertTrue(torch.equal(mask.cpu(), min_p_keep_mask(logits, min_p).cpu()))
        self.assertTrue(bool(mask[0, 0]))
        self.assertTrue(bool(mask[1, 1]))


if __name__ == "__main__":
    unittest.main()
