#!/usr/bin/env python3
"""Framework-independent exact tokenwise Recirculation oracle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from tllm.consumers.recirculation.consumer import RecirculationConsumer


class ReferenceDecoder(Protocol):
    @property
    def layer_count(self) -> int: ...

    def embed(self, token_id: int, position: int) -> torch.Tensor: ...

    def run_layer(
        self,
        layer: int,
        hidden: torch.Tensor,
        *,
        position: int,
        overwrite_kv: bool,
    ) -> torch.Tensor: ...

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class SerialRecirculationResult:
    logits: tuple[torch.Tensor, ...]
    recurrent_states: tuple[torch.Tensor, ...]


def run_serial_recirculation(
    decoder: ReferenceDecoder,
    consumer: RecirculationConsumer,
    token_ids: tuple[int, ...],
) -> SerialRecirculationResult:
    """Run the exact normal-logits/recurrent-KV ordering one token at a time."""

    source_layer, destination_layer = consumer.config.resolved_layers(decoder.layer_count)
    logits: list[torch.Tensor] = []
    recurrent_states: list[torch.Tensor] = []
    for position, token_id in enumerate(token_ids):
        hidden = decoder.embed(int(token_id), position)
        source: torch.Tensor | None = None
        destination: torch.Tensor | None = None
        for layer in range(decoder.layer_count):
            hidden = decoder.run_layer(
                layer,
                hidden,
                position=position,
                overwrite_kv=False,
            )
            if layer == destination_layer:
                destination = hidden
            if layer == source_layer:
                source = hidden
        assert source is not None and destination is not None
        # The observable logits are from the normal pass.  The replay exists
        # solely to publish the upper-layer state consumed by later tokens.
        logits.append(decoder.project_logits(hidden))
        recurrent = consumer.mix(
            source,
            destination,
            torch.tensor([position], device=destination.device),
        )
        for layer in range(destination_layer + 1, decoder.layer_count):
            recurrent = decoder.run_layer(
                layer,
                recurrent,
                position=position,
                overwrite_kv=True,
            )
        recurrent_states.append(recurrent.detach().clone())
    return SerialRecirculationResult(tuple(logits), tuple(recurrent_states))


__all__ = [
    "ReferenceDecoder",
    "SerialRecirculationResult",
    "run_serial_recirculation",
]
