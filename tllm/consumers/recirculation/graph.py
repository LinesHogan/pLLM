#!/usr/bin/env python3
"""Compatibility facade for the backend-independent Recirculation graph."""

from __future__ import annotations

from tllm.consumers.recirculation.config import RecirculationConfig
from tllm.contracts.recirculation import RecirculationPrefillBindingConfig
from tllm.workgraph.recirculation import build_exact_recirculation_graph
from tllm.workgraph.recurrence import RecurrenceGraph


def build_recirculation_graph(
    config: RecirculationConfig,
    *,
    token_count: int,
    layer_count: int,
) -> RecurrenceGraph:
    return build_exact_recirculation_graph(
        RecirculationPrefillBindingConfig(
            source_layer=config.source_layer,
            destination_layer=config.destination_layer,
            alpha=config.alpha,
            beta=config.beta,
            ramp_tokens=config.ramp_tokens,
        ),
        token_count=token_count,
        layer_count=layer_count,
    )


__all__ = ["build_recirculation_graph"]
