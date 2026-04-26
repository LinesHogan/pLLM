#!/usr/bin/env python3
"""Unit tests for devlist consumer shared primitives."""

from __future__ import annotations

import unittest

import torch


class DevlistConsumerSurfaceUnitTest(unittest.TestCase):
    def test_package_exports_low_rank_adapter(self) -> None:
        from tllm.consumers.devlist import LowRankAdapter

        adapter = LowRankAdapter(in_dim=4, out_dim=6, rank=2)

        self.assertEqual(adapter.rank, 2)

    def test_low_rank_adapter_zero_b_starts_as_noop(self) -> None:
        from tllm.consumers.devlist import LowRankAdapter

        adapter = LowRankAdapter(in_dim=4, out_dim=6, rank=2)
        x = torch.randn(3, 4)

        out = adapter(x)

        self.assertEqual(tuple(out.shape), (3, 6))
        self.assertTrue(torch.allclose(out, torch.zeros_like(out)))

    def test_low_rank_adapter_updates_after_optimizer_step(self) -> None:
        from tllm.consumers.devlist import LowRankAdapter

        adapter = LowRankAdapter(in_dim=4, out_dim=6, rank=2)
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
        x = torch.randn(3, 4)
        target = torch.randn(3, 6)

        before = adapter(x).detach().clone()
        loss = torch.nn.functional.mse_loss(adapter(x), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after = adapter(x).detach()

        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
