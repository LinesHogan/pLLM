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
    ttt_proj_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor, int, int]:
    if past_h is not None:
        assert past_t is not None
        h = torch.cat([past_h, h], dim=0)
        t = torch.cat([past_t, t], dim=0)
    fast_w = down_weight.clone() if past_w is None else past_w.clone()
    chunk = max(1, int(chunk_size))
    full_rows = (int(h.shape[0]) // chunk) * chunk
    outputs: list[torch.Tensor] = []
    updates = 0
    for start in range(0, full_rows, chunk):
        h_chunk = h[start : start + chunk]
        t_chunk = t[start : start + chunk]
        outputs.append(h_chunk.matmul(fast_w.t()))
        if ttt_proj_weight is None:
            dw = torch.einsum("ch,cd->dh", h_chunk, t_chunk) * float(learning_rate)
        else:
            dw = torch.einsum("ch,cd,de->eh", h_chunk, t_chunk, ttt_proj_weight) * float(learning_rate)
        fast_w = fast_w + dw
        updates += 1
    tail_h = h[full_rows:]
    tail_t = t[full_rows:]
    if int(tail_h.shape[0]) > 0:
        outputs.append(tail_h.matmul(fast_w.t()))
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
    def test_exact_toy_upstream_proj_formula_outputs_before_update(self) -> None:
        from tllm.consumers.inplace_ttt import run_exact_toy_down_proj_step

        down_weight = torch.tensor(
            [[0.2, -0.3, 0.5], [0.4, 0.1, -0.2]],
            dtype=torch.float64,
        )
        h = torch.tensor([[1.0, -0.5, 0.25], [0.2, 0.4, -0.3]], dtype=torch.float64)
        t = torch.tensor([[0.5, -0.25], [0.1, 0.75]], dtype=torch.float64)
        ttt_proj_weight = torch.tensor([[0.9, -0.1], [0.2, 0.7]], dtype=torch.float64)

        expected_output, _tail_h, _tail_t, expected_w, updates, trained_rows = _reference_exact_step(
            h,
            t,
            down_weight,
            chunk_size=2,
            learning_rate=0.3,
            ttt_proj_weight=ttt_proj_weight,
        )

        result = run_exact_toy_down_proj_step(
            h,
            t,
            down_weight,
            chunk_size=2,
            learning_rate=0.3,
            ttt_proj_weight=ttt_proj_weight,
        )

        self.assertTrue(torch.allclose(result.output, expected_output))
        self.assertTrue(torch.allclose(result.state.past_w, expected_w))
        self.assertEqual(result.update_count, updates)
        self.assertEqual(result.trained_rows, trained_rows)


if __name__ == "__main__":
    unittest.main()
