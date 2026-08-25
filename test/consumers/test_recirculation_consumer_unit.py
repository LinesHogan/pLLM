#!/usr/bin/env python3
"""Pure Recirculation declaration and mixing math."""

from __future__ import annotations

import pytest
import torch

from tllm.consumers.recirculation import RecirculationConfig, RecirculationConsumer
from tllm.consumers.recirculation import run_serial_recirculation


def test_mix_matches_norm_and_ramp() -> None:
    consumer = RecirculationConsumer(
        RecirculationConfig(
            source_layer=2,
            destination_layer=0,
            alpha=0.2,
            ramp_tokens=10,
        )
    )
    source = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    destination = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    actual = consumer.mix(source, destination, torch.tensor([0, 5, 10]))
    torch.testing.assert_close(
        actual,
        torch.tensor([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]]),
    )


def test_negative_layers_resolve_before_graph_construction() -> None:
    config = RecirculationConfig(source_layer=-1, destination_layer=-3)
    assert config.resolved_layers(4) == (3, 1)
    graph = RecirculationConsumer(config).recurrence_graph(token_count=1, layer_count=4)
    assert any(op.op_id == "t0/recurrent/l2" for op in graph.ops)


def test_invalid_layer_order_fails_before_execution() -> None:
    consumer = RecirculationConsumer(
        RecirculationConfig(source_layer=1, destination_layer=1)
    )
    with pytest.raises(ValueError, match="destination_layer"):
        consumer.recurrence_graph(token_count=1, layer_count=4)


class _TinyCausalDecoder:
    layer_count = 3

    def __init__(self) -> None:
        self.kv: dict[tuple[int, int], torch.Tensor] = {}
        self.calls: list[tuple[int, int, bool]] = []

    def embed(self, token_id: int, position: int) -> torch.Tensor:
        del position
        return torch.tensor([[float(token_id), 1.0]])

    def run_layer(
        self,
        layer: int,
        hidden: torch.Tensor,
        *,
        position: int,
        overwrite_kv: bool,
    ) -> torch.Tensor:
        self.calls.append((position, layer, overwrite_kv))
        previous = sum(
            (
                value
                for (cached_layer, cached_position), value in self.kv.items()
                if cached_layer == layer and cached_position < position
            ),
            torch.zeros_like(hidden),
        )
        out = hidden + previous + float(layer + 1)
        self.kv[(layer, position)] = out.detach().clone()
        return out

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.sum(dim=-1)


def test_serial_oracle_returns_normal_logits_and_overwrites_upper_kv() -> None:
    decoder = _TinyCausalDecoder()
    consumer = RecirculationConsumer(
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2)
    )
    result = run_serial_recirculation(decoder, consumer, (1, 2))

    assert len(result.logits) == 2
    assert decoder.calls == [
        (0, 0, False),
        (0, 1, False),
        (0, 2, False),
        (0, 1, True),
        (0, 2, True),
        (1, 0, False),
        (1, 1, False),
        (1, 2, False),
        (1, 1, True),
        (1, 2, True),
    ]
    # The cache exposed to token 1 is the replayed token-0 upper state, not
    # token 0's normal upper state.
    assert torch.equal(decoder.kv[(2, 0)], result.recurrent_states[0])
