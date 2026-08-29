#!/usr/bin/env python3
"""Small Recirculation consumer: configuration, recurrence, and mix math only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tllm.consumers.base import BaseConsumer
from tllm.consumers.recirculation.config import LegacyRecirculationConfig
from tllm.consumers.recirculation.graph import build_recirculation_graph
from tllm.ports.base import ConsumerFlow
from tllm.workgraph.recurrence import (
    DependenceKind,
    RecurrenceGraph,
    RecurrenceOp,
    RecurrenceRequirements,
)


@dataclass(frozen=True, slots=True)
class _RecirculationRuntimeAdapter:
    consumer: "RecirculationConsumer"
    source_layer: int
    destination_layer: int

    def execute(
        self,
        op: RecurrenceOp,
        predecessors: tuple[tuple[RecurrenceOp, torch.Tensor], ...],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if op.executor != "recirculation_mix" or op.point.pass_name != "mix":
            raise RuntimeError(
                f"Recirculation cannot execute recurrence op {op.op_id!r}"
            )
        source = next(
            (
                value
                for predecessor, value in predecessors
                if predecessor.point.pass_name == "normal"
                and predecessor.point.layer == self.source_layer
            ),
            None,
        )
        destination = next(
            (
                value
                for predecessor, value in predecessors
                if predecessor.point.pass_name == "normal"
                and predecessor.point.layer == self.destination_layer
            ),
            None,
        )
        if source is None or destination is None:
            raise RuntimeError(
                "Recirculation mix requires normal source and destination states"
            )
        return self.consumer.mix(source, destination, positions)


class RecirculationConsumer(BaseConsumer[LegacyRecirculationConfig]):
    def __init__(self, config: LegacyRecirculationConfig) -> None:
        self.config = config

    @property
    def consumer_id(self) -> str:
        return self.config.consumer_id

    def flows(self) -> Sequence[ConsumerFlow]:
        # Execution is represented by recurrence_graph(), not a post-forward
        # bundle.  Returning a fake residual flow would lose carried versions.
        return ()

    def recurrence_requirements(self) -> RecurrenceRequirements | None:
        if not self.config.enabled:
            return None
        return RecurrenceRequirements(
            phase="prefill",
            effects=frozenset(
                {
                    DependenceKind.DATA,
                    DependenceKind.CONTROL,
                    DependenceKind.KV_WRITE_READ,
                    DependenceKind.KV_OVERWRITE_READ,
                }
            ),
            jacobi=self.config.jacobi,
            jacobi_carried_kinds=frozenset(
                {DependenceKind.KV_OVERWRITE_READ}
            ),
        )

    def recurrence_graph(self, *, token_count: int, layer_count: int) -> RecurrenceGraph:
        return build_recirculation_graph(
            self.config,
            token_count=token_count,
            layer_count=layer_count,
        )

    def recurrence_runtime_adapter(self, *, layer_count: int) -> object:
        source, destination = self.config.resolved_layers(layer_count)
        return _RecirculationRuntimeAdapter(self, source, destination)

    def mix(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if source.shape != destination.shape:
            raise RuntimeError("Recirculation source and destination shapes must match")
        if source.ndim < 2:
            raise RuntimeError("Recirculation residuals must have a hidden dimension")
        flat_positions = positions[0] if positions.ndim == 2 else positions
        if flat_positions.ndim != 1 or int(flat_positions.shape[0]) != int(source.shape[0]):
            raise RuntimeError("Recirculation positions must align with residual rows")
        source_float = source.float()
        destination_float = destination.float()
        source_norm = torch.linalg.vector_norm(source_float, dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        destination_norm = torch.linalg.vector_norm(destination_float, dim=-1, keepdim=True)
        normalized = source_float * (destination_norm / source_norm)
        alpha: float | torch.Tensor = float(self.config.alpha)
        if self.config.ramp_tokens:
            ramp = flat_positions.to(dtype=torch.float32).clamp(
                min=0, max=int(self.config.ramp_tokens)
            )
            alpha = (ramp / int(self.config.ramp_tokens)).unsqueeze(-1) * float(
                self.config.alpha
            )
        beta = (1.0 - alpha) if self.config.beta is None else float(self.config.beta)
        return (beta * destination_float + alpha * normalized).to(destination.dtype)


__all__ = ["RecirculationConsumer"]
