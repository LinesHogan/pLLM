"""Recirculation-specific residual mixing math and legacy adapter."""

from __future__ import annotations

import torch

from tllm.consumers.recirculation.config import RecirculationConfig
from tllm.workgraph.recurrence import RecurrenceOp


def mix_recirculation_tensors(
    config: RecirculationConfig,
    source: torch.Tensor,
    destination: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    flat_positions = positions[0] if positions.ndim == 2 else positions
    source_float = source.float()
    destination_float = destination.float()
    source_norm = torch.linalg.vector_norm(
        source_float, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float32).tiny)
    destination_norm = torch.linalg.vector_norm(
        destination_float, dim=-1, keepdim=True
    )
    normalized = source_float * (destination_norm / source_norm)
    alpha: float | torch.Tensor = config.alpha
    if config.ramp_tokens:
        ramp = flat_positions.float().clamp(0, config.ramp_tokens)
        alpha = (ramp / config.ramp_tokens).unsqueeze(-1) * config.alpha
    beta = 1.0 - alpha if config.beta is None else config.beta
    return (beta * destination_float + alpha * normalized).to(destination.dtype)


class RecirculationMixAdapter:
    def __init__(self, config: RecirculationConfig, *, layer_count: int) -> None:
        self.config = config
        self.source_layer, self.destination_layer = config.resolved_layers(layer_count)

    def execute(
        self,
        op: RecurrenceOp,
        predecessors: tuple[tuple[RecurrenceOp, torch.Tensor], ...],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        source = next(
            value
            for predecessor, value in predecessors
            if predecessor.point.pass_name == "normal"
            and predecessor.point.layer == self.source_layer
        )
        destination = next(
            value
            for predecessor, value in predecessors
            if predecessor.point.pass_name == "normal"
            and predecessor.point.layer == self.destination_layer
        )
        return mix_recirculation_tensors(
            self.config, source, destination, positions
        )


__all__ = ["RecirculationMixAdapter", "mix_recirculation_tensors"]
