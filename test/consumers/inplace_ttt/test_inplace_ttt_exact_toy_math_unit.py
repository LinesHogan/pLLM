#!/usr/bin/env python3
"""Exact toy-math tests for In-Place TTT chunk/cache behavior."""

from __future__ import annotations

import unittest

import torch


def _reference_exact_step(
    h: torch.Tensor,
    t: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    past_h: torch.Tensor | None = None,
    past_t: torch.Tensor | None = None,
    past_w: torch.Tensor | None = None,
    chunk_size: int,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, int, int]:
    if past_h is not None:
        assert past_t is not None
        h = torch.cat([past_h, h], dim=0)
        t = torch.cat([past_t, t], dim=0)
    fast_w = torch.zeros_like(down_weight) if past_w is None else past_w.clone()
    chunk = max(1, int(chunk_size))
    full_rows = (int(h.shape[0]) // chunk) * chunk
    outputs: list[torch.Tensor] = []
    updates = 0
    for start in range(0, full_rows, chunk):
        h_chunk = h[start : start + chunk]
        t_chunk = t[start : start + chunk]
        effective_before = down_weight + fast_w
        residual = t_chunk - h_chunk.matmul(effective_before.t())
        fast_w = fast_w + float(learning_rate) * residual.t().matmul(h_chunk) / float(chunk)
        outputs.append(h_chunk.matmul((down_weight + fast_w).t()))
        updates += 1
    tail_h = h[full_rows:]
    tail_t = t[full_rows:]
    if int(tail_h.shape[0]) > 0:
        outputs.append(tail_h.matmul((down_weight + fast_w).t()))
    output = torch.cat(outputs, dim=0) if outputs else h.new_empty((0, down_weight.shape[0]))
    return (
        output,
        tail_h.clone() if int(tail_h.shape[0]) > 0 else None,
        tail_t.clone() if int(tail_t.shape[0]) > 0 else None,
        fast_w,
        updates,
        full_rows,
    )


class InPlaceTTTExactToyMathUnitTest(unittest.TestCase):
    def test_exact_toy_full_chunks_match_direct_reference(self) -> None:
        from tllm.consumers.inplace_ttt import run_exact_toy_down_proj_step

        down_weight = torch.tensor(
            [[0.30, -0.20, 0.10], [-0.40, 0.25, 0.50]],
            dtype=torch.float64,
        )
        h = torch.tensor(
            [
                [1.0, -0.5, 0.25],
                [0.2, 0.4, -0.3],
                [-0.6, 0.8, 0.5],
                [0.7, -0.1, -0.4],
            ],
            dtype=torch.float64,
        )
        t = torch.tensor(
            [
                [0.2, -0.1],
                [0.4, 0.6],
                [-0.3, 0.7],
                [0.9, -0.2],
            ],
            dtype=torch.float64,
        )

        expected_output, expected_tail_h, expected_tail_t, expected_w, updates, trained_rows = _reference_exact_step(
            h,
            t,
            down_weight,
            chunk_size=2,
            learning_rate=0.15,
        )

        result = run_exact_toy_down_proj_step(
            h,
            t,
            down_weight,
            chunk_size=2,
            learning_rate=0.15,
        )

        self.assertTrue(torch.allclose(result.output, expected_output))
        self.assertIsNone(expected_tail_h)
        self.assertIsNone(expected_tail_t)
        self.assertIsNone(result.state.past_h)
        self.assertIsNone(result.state.past_t)
        self.assertTrue(torch.allclose(result.state.past_w, expected_w))
        self.assertEqual(result.update_count, updates)
        self.assertEqual(result.trained_rows, trained_rows)
        self.assertEqual(result.tail_rows, 0)

    def test_exact_toy_below_chunk_only_caches_tail_without_update(self) -> None:
        from tllm.consumers.inplace_ttt import run_exact_toy_down_proj_step

        down_weight = torch.tensor([[0.25, -0.5], [0.75, 0.10]], dtype=torch.float64)
        h = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]], dtype=torch.float64)
        t = torch.tensor([[0.5, 1.0], [-0.25, 0.75], [0.0, -0.5]], dtype=torch.float64)

        result = run_exact_toy_down_proj_step(
            h,
            t,
            down_weight,
            chunk_size=4,
            learning_rate=0.2,
        )

        self.assertTrue(torch.allclose(result.output, h.matmul(down_weight.t())))
        self.assertTrue(torch.equal(result.state.past_h, h))
        self.assertTrue(torch.equal(result.state.past_t, t))
        self.assertTrue(torch.equal(result.state.past_w, torch.zeros_like(down_weight)))
        self.assertEqual(result.update_count, 0)
        self.assertEqual(result.trained_rows, 0)
        self.assertEqual(result.tail_rows, 3)

    def test_exact_toy_cached_tail_forms_next_chunk_and_keeps_new_tail(self) -> None:
        from tllm.consumers.inplace_ttt import run_exact_toy_down_proj_step

        down_weight = torch.tensor(
            [[0.1, -0.2, 0.3], [0.4, 0.0, -0.1]],
            dtype=torch.float64,
        )
        first_h = torch.tensor([[0.5, -1.0, 0.25]], dtype=torch.float64)
        first_t = torch.tensor([[0.2, -0.4]], dtype=torch.float64)
        next_h = torch.tensor(
            [
                [1.0, 0.5, -0.5],
                [-0.5, 0.75, 0.25],
                [0.3, -0.2, 0.9],
                [0.8, 0.1, -0.6],
            ],
            dtype=torch.float64,
        )
        next_t = torch.tensor(
            [
                [0.7, -0.1],
                [-0.2, 0.5],
                [0.4, 0.4],
                [-0.3, 0.2],
            ],
            dtype=torch.float64,
        )
        chunk_size = 3
        learning_rate = 0.1

        first = run_exact_toy_down_proj_step(
            first_h,
            first_t,
            down_weight,
            chunk_size=chunk_size,
            learning_rate=learning_rate,
        )
        expected_output, expected_tail_h, expected_tail_t, expected_w, updates, trained_rows = _reference_exact_step(
            next_h,
            next_t,
            down_weight,
            past_h=first_h,
            past_t=first_t,
            past_w=torch.zeros_like(down_weight),
            chunk_size=chunk_size,
            learning_rate=learning_rate,
        )

        second = run_exact_toy_down_proj_step(
            next_h,
            next_t,
            down_weight,
            state=first.state,
            chunk_size=chunk_size,
            learning_rate=learning_rate,
        )

        self.assertTrue(torch.allclose(second.output, expected_output))
        self.assertTrue(torch.equal(second.state.past_h, expected_tail_h))
        self.assertTrue(torch.equal(second.state.past_t, expected_tail_t))
        self.assertTrue(torch.allclose(second.state.past_w, expected_w))
        self.assertEqual(second.update_count, updates)
        self.assertEqual(second.trained_rows, trained_rows)
        self.assertEqual(second.tail_rows, 2)


if __name__ == "__main__":
    unittest.main()
