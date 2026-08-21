#!/usr/bin/env python3
"""Fixed toy parity tests for the OverRIDE consumer core."""

from __future__ import annotations

import unittest

import torch
from torch import nn


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


_DOWN_WEIGHTS = (
    torch.tensor([[0.5, -1.0], [1.5, 0.25]], dtype=torch.float32),
    torch.tensor([[1.0, 0.5], [-0.25, 2.0]], dtype=torch.float32),
)
_UP_WEIGHTS = (
    torch.tensor([[2.0, -0.5], [0.25, 1.0], [-1.0, 0.75]], dtype=torch.float32),
    torch.tensor([[0.5, 1.25], [-1.5, 0.5], [0.75, -0.25]], dtype=torch.float32),
)


def _set_fixed_weights(consumer: object, reference: _ToyOverRIDEReference) -> None:
    from tllm.consumers.override import OverRIDEConsumer

    assert isinstance(consumer, OverRIDEConsumer)
    # Warm up so the program is compiled and the state bank exists.
    hidden = torch.zeros((2, 2), dtype=torch.float32)
    logits = torch.zeros((2, 3), dtype=torch.float32)
    consumer.reweight_logits(hidden, logits, iteration=2)
    compiled = consumer.compiled
    assert compiled is not None
    params = compiled.bank("heads").slot_for(None).params
    with torch.no_grad():
        for idx, (down_weight, up_weight) in enumerate(zip(_DOWN_WEIGHTS, _UP_WEIGHTS, strict=True)):
            params["down"][idx].copy_(down_weight)
            params["up"][idx].copy_(up_weight)
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

        reweighted = consumer.reweight_logits(hidden, logits, iteration=2)
        expected_logits, _expected_q = reference.compute(hidden, logits, iteration=2, lambd=config.lambd)
        explicit_residual = reference.heads[0](hidden.detach())

        self.assertTrue(torch.equal(expected_logits, logits - config.lambd * explicit_residual))
        self.assertEqual(float((reweighted - expected_logits).abs().max()), 0.0)

    def test_batch_reweight_then_update_trains_bank_params(self) -> None:
        from tllm.consumers.override import OverRIDEConsumer, OverRIDEConsumerConfig

        torch.manual_seed(0)
        consumer = OverRIDEConsumer(
            OverRIDEConsumerConfig(
                hidden_size=2,
                vocab_size=3,
                rank=2,
                num_iterations=3,
                lambd=0.75,
                learning_rate=0.2,
            )
        )
        _set_fixed_weights(consumer, _ToyOverRIDEReference(
            hidden_size=2, vocab_size=3, rank=2, num_iterations=3, learning_rate=0.2
        ))
        compiled = consumer.compiled
        assert compiled is not None
        up_before = compiled.bank("heads").slot_for(None).params["up"].clone()

        hidden = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)
        logits = torch.tensor([[10.0, 20.0, 30.0], [1.5, -2.0, 4.0]], dtype=torch.float32)
        consumer.reweight_logits_batch(hidden, logits, iterations=(2, 2))
        consumer.update_from_sampled_tokens(torch.tensor([0, 2], dtype=torch.long))

        stats = consumer.read_stats()
        self.assertEqual(stats.update_count, 1)
        self.assertEqual(stats.loss_count, 1)
        self.assertGreater(stats.last_loss, 0.0)
        up_after = compiled.bank("heads").slot_for(None).params["up"]
        # Only the current-iteration head (index 1) receives gradient.
        self.assertFalse(torch.equal(up_after[1], up_before[1]))
        self.assertTrue(torch.equal(up_after[0], up_before[0]))


if __name__ == "__main__":
    unittest.main()
