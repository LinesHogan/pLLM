#!/usr/bin/env python3
"""Configuration for residual-stream Recirculation."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.workgraph.recurrence import JacobiPolicy


@dataclass(frozen=True, slots=True)
class RecirculationConfig:
    source_layer: int = -1
    destination_layer: int = 0
    alpha: float = 0.15
    beta: float | None = None
    ramp_tokens: int = 0
    jacobi: JacobiPolicy | None = None
    consumer_id: str = "recirculation"
    enabled: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.source_layer, bool) or not isinstance(self.source_layer, int):
            raise TypeError("source_layer must be int")
        if isinstance(self.destination_layer, bool) or not isinstance(self.destination_layer, int):
            raise TypeError("destination_layer must be int")
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if self.beta is not None and not 0.0 <= float(self.beta) <= 1.0:
            raise ValueError("beta must be between 0 and 1")
        if self.ramp_tokens < 0:
            raise ValueError("ramp_tokens must be non-negative")
        if self.jacobi is not None and not isinstance(self.jacobi, JacobiPolicy):
            raise TypeError("jacobi must be JacobiPolicy or None")
        if not self.consumer_id:
            raise ValueError("consumer_id must be non-empty")

    def resolved_layers(self, layer_count: int) -> tuple[int, int]:
        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        source = self.source_layer if self.source_layer >= 0 else layer_count + self.source_layer
        destination = (
            self.destination_layer
            if self.destination_layer >= 0
            else layer_count + self.destination_layer
        )
        if not 0 <= destination < source < layer_count:
            raise ValueError(
                "Recirculation layers must satisfy "
                "0 <= destination_layer < source_layer < layer_count"
            )
        return source, destination


__all__ = ["RecirculationConfig"]
