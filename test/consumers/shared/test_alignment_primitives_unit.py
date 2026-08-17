#!/usr/bin/env python3
"""Unit tests for shared consumer alignment helpers."""

from __future__ import annotations

import math
import unittest

import torch


class DevlistAlignmentPrimitivesUnitTest(unittest.TestCase):
    def test_sentence_cross_entropy_selects_high_ce_rows(self) -> None:
        from tllm.consumers._common.alignment import select_rows_above_threshold, sentence_cross_entropy

        logits = torch.tensor(
            [
                [[4.0, 0.0], [4.0, 0.0]],
                [[0.0, 4.0], [0.0, 4.0]],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)

        ce = sentence_cross_entropy(logits, labels)
        selected = select_rows_above_threshold(ce, threshold=float(ce.mean()))

        self.assertEqual(selected, [1])

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

    def test_select_top_k_per_prompt_matches_vote_sample_ordering(self) -> None:
        from tllm.consumers._common.alignment import select_top_k_per_prompt

        selected = select_top_k_per_prompt(num_prompts=2, votes_per_prompt=4, samples_per_prompt=2)

        self.assertEqual(selected, [0, 1, 4, 5])


if __name__ == "__main__":
    unittest.main()
