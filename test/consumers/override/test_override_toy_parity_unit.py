#!/usr/bin/env python3
"""Fixed toy parity tests for the OverRIDE consumer core."""

from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F


class _ToyLowRankHead(nn.Module):
    def __init__(self, *, hidden_size: int, vocab_size: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden))


class _ToyOverRIDEReference(nn.Module):
    """Small local reference for the upstream OverRIDE sampler-side math."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        num_iterations: int,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                _ToyLowRankHead(hidden_size=hidden_size, vocab_size=vocab_size, rank=rank)
                for _ in range(max(0, int(num_iterations) - 1))
            ]
        )
        self.optimizer = torch.optim.Adam(self.heads.parameters(), lr=float(learning_rate))

    def _head_for_iteration(self, iteration: int) -> _ToyLowRankHead | None:
        idx = int(iteration) - 1
        if idx < 0 or idx >= len(self.heads):
            return None
        return self.heads[idx]

    def compute(
        self,
        hidden: torch.Tensor,
        logits: torch.Tensor,
        *,
        iteration: int,
        lambd: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_for_head = hidden.detach()
        logits_for_update = logits.detach()
        previous_head = self._head_for_iteration(int(iteration) - 1)
        current_head = self._head_for_iteration(int(iteration))

        reweighted = logits
        if previous_head is not None:
            residual = previous_head(hidden_for_head).to(dtype=logits.dtype)
            reweighted = logits - float(lambd) * residual

        q_logits_next = None
        if current_head is not None:
            q_logits_next = logits_for_update + current_head(hidden_for_head).to(dtype=logits.dtype)
        return reweighted, q_logits_next

    def update(self, q_logits_next: torch.Tensor, sampled_token_ids: torch.Tensor) -> None:
        loss = F.cross_entropy(
            q_logits_next.reshape(-1, int(q_logits_next.shape[-1])),
            sampled_token_ids.reshape(-1).to(device=q_logits_next.device, dtype=torch.long),
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def _set_fixed_weights(consumer: object, reference: _ToyOverRIDEReference) -> None:
    heads = getattr(consumer, "engine").heads
    down_weights = (
        torch.tensor([[0.5, -1.0], [1.5, 0.25]], dtype=torch.float32),
        torch.tensor([[1.0, 0.5], [-0.25, 2.0]], dtype=torch.float32),
    )
    up_weights = (
        torch.tensor([[2.0, -0.5], [0.25, 1.0], [-1.0, 0.75]], dtype=torch.float32),
        torch.tensor([[0.5, 1.25], [-1.5, 0.5], [0.75, -0.25]], dtype=torch.float32),
    )
    with torch.no_grad():
        for idx, (down_weight, up_weight) in enumerate(zip(down_weights, up_weights, strict=True)):
            heads[idx].down.weight.copy_(down_weight)
            heads[idx].up.weight.copy_(up_weight)
            reference.heads[idx].down.weight.copy_(down_weight)
            reference.heads[idx].up.weight.copy_(up_weight)


class OverRIDEToyParityUnitTest(unittest.TestCase):
    def test_toy_residual_reweight_matches_reference_formula_with_zero_diff(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        config = OverRIDEConsumerConfig(
            hidden_size=2,
            vocab_size=3,
            rank=2,
            num_iterations=3,
            lambd=0.75,
            learning_rate=0.05,
        )
        consumer = OverRIDEConsumer(config)
        reference = _ToyOverRIDEReference(
            hidden_size=2,
            vocab_size=3,
            rank=2,
            num_iterations=3,
            learning_rate=0.05,
        )
        _set_fixed_weights(consumer, reference)
        hidden = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)
        logits = torch.tensor([[10.0, 20.0, 30.0], [1.5, -2.0, 4.0]], dtype=torch.float32)

        result = consumer.reweight_logits(hidden, logits, iteration=2)
        expected_logits, _expected_q = reference.compute(hidden, logits, iteration=2, lambd=config.lambd)
        explicit_residual = consumer.engine.heads[0](hidden.detach())

        self.assertTrue(torch.equal(expected_logits, logits - config.lambd * explicit_residual))
        self.assertEqual(float((result.logits - expected_logits).abs().max()), 0.0)

    def test_toy_q_logits_next_and_sampled_token_update_match_reference_with_zero_diff(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        config = OverRIDEConsumerConfig(
            hidden_size=2,
            vocab_size=3,
            rank=2,
            num_iterations=3,
            lambd=0.5,
            learning_rate=0.05,
        )
        consumer = OverRIDEConsumer(config)
        reference = _ToyOverRIDEReference(
            hidden_size=2,
            vocab_size=3,
            rank=2,
            num_iterations=3,
            learning_rate=0.05,
        )
        _set_fixed_weights(consumer, reference)
        hidden = torch.tensor([[1.0, -2.0], [3.0, 0.5]], dtype=torch.float32)
        logits = torch.tensor([[4.0, 0.5, -1.0], [2.0, 3.5, 1.0]], dtype=torch.float32)
        sampled_token_ids = torch.tensor([2, 0])

        result = consumer.reweight_logits(hidden, logits, iteration=2)
        _expected_logits, expected_q = reference.compute(hidden, logits, iteration=2, lambd=config.lambd)
        assert result.q_logits_next is not None
        assert expected_q is not None

        self.assertEqual(float((result.q_logits_next - expected_q).abs().max()), 0.0)

        consumer.update_from_sampled_tokens(result.q_logits_next, sampled_token_ids)
        reference.update(expected_q, sampled_token_ids)

        for local_head, reference_head in zip(consumer.engine.heads, reference.heads, strict=True):
            self.assertEqual(float((local_head.down.weight - reference_head.down.weight).abs().max()), 0.0)
            self.assertEqual(float((local_head.up.weight - reference_head.up.weight).abs().max()), 0.0)

    def test_row_id_boundary_rejects_ambiguous_types_and_caches_normalized_ints(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=3,
                learning_rate=0.1,
            )
        )
        hidden = torch.tensor([[1.0, -1.0], [0.5, 2.0]], dtype=torch.float32)
        logits = torch.randn(2, 3)

        for bad_row_ids in ((True, 2), (1.0, 2.0)):
            with self.subTest(row_ids=bad_row_ids):
                with self.assertRaisesRegex(RuntimeError, "row ids must be integers"):
                    consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2), row_ids=bad_row_ids)

        with self.assertRaisesRegex(RuntimeError, "row ids must be a tuple of integers"):
            consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2), row_ids=b"12")

        for bad_tensor in (
            torch.tensor([[1, 2]], dtype=torch.long),
            torch.tensor([True, False]),
            torch.tensor([1.0, 2.0]),
        ):
            with self.subTest(dtype=str(bad_tensor.dtype), shape=tuple(bad_tensor.shape)):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "row id tensors must be 1D|row id tensors must use an integer dtype",
                ):
                    consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2), row_ids=bad_tensor)

        row_ids = torch.tensor([21, 22], dtype=torch.long)
        consumer.reweight_logits_batch(hidden, logits, iterations=(1, 2), row_ids=row_ids)
        row_ids.fill_(99)
        before = consumer.engine.heads[0].up.weight.detach().clone()

        consumer.update_from_sampled_tokens_batch(torch.tensor([0, 1]), row_ids=(21, 22))

        self.assertFalse(torch.equal(before, consumer.engine.heads[0].up.weight.detach()))


if __name__ == "__main__":
    unittest.main()
