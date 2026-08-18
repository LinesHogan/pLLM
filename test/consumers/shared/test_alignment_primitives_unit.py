#!/usr/bin/env python3
"""Unit tests for shared consumer alignment helpers."""

from __future__ import annotations

import math
import unittest

import torch


class SharedAlignmentPrimitivesUnitTest(unittest.TestCase):
    def test_detached_train_input_reuses_normal_storage(self) -> None:
        from tllm.consumers._common.alignment import detached_train_input

        source = torch.randn(2, 3, requires_grad=True)
        result = detached_train_input(source)

        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.requires_grad)

    def test_detached_train_input_clones_inference_storage(self) -> None:
        from tllm.consumers._common.alignment import detached_train_input

        with torch.inference_mode():
            source = torch.randn(2, 3)
        result = detached_train_input(source)

        self.assertNotEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(torch.is_inference(result))

    def test_weighted_one_hot_kl_matches_expected_ce_weighting(self) -> None:
        from tllm.consumers._common.alignment import weighted_one_hot_kl_loss

        logits = torch.tensor(
            [
                [[2.0, 0.0], [2.0, 0.0]],
                [[0.0, 2.0], [0.0, 2.0]],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)
        sentence_ce = torch.tensor([0.12692805, 2.126928], dtype=torch.float32)
        threshold = 1.0
        lamb = 0.5

        loss = weighted_one_hot_kl_loss(logits, labels, sentence_ce=sentence_ce, threshold=threshold, lamb=lamb)

        expected_weight = lamb * math.exp(float(sentence_ce[1] - threshold))
        self.assertAlmostEqual(float(loss), float(sentence_ce[1]) * expected_weight, places=5)

    def test_extract_boxed_answer_normalizes_equivalent_answers(self) -> None:
        from tllm.consumers._common.alignment import normalize_answer_text

        left = normalize_answer_text("Long reasoning. Therefore \\\\boxed{ 4 }")
        right = normalize_answer_text("Short reasoning. Final: \\\\boxed{4}")

        self.assertEqual(left, "4")
        self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
