"""Tensor resource binding for exact Recirculation recurrence operations."""

from __future__ import annotations

from typing import Protocol

import torch

from tllm.workgraph.recurrence import RecurrenceOp


class RecirculationMixConfig(Protocol):
    @property
    def alpha(self) -> float: ...

    @property
    def beta(self) -> float | None: ...

    @property
    def ramp_tokens(self) -> int: ...

    def resolved_layers(self, layer_count: int) -> tuple[int, int]: ...


def mix_recirculation_tensors(
    config: RecirculationMixConfig,
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
    source_norm = torch.linalg.vector_norm(
        source_float, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(torch.float32).tiny)
    destination_norm = torch.linalg.vector_norm(
        destination_float, dim=-1, keepdim=True
    )
    normalized = source_float * (destination_norm / source_norm)
    alpha: float | torch.Tensor = float(config.alpha)
    if config.ramp_tokens:
        ramp = flat_positions.to(dtype=torch.float32).clamp(
            min=0, max=int(config.ramp_tokens)
        )
        alpha = (ramp / int(config.ramp_tokens)).unsqueeze(-1) * float(config.alpha)
    beta = (1.0 - alpha) if config.beta is None else float(config.beta)
    return (beta * destination_float + alpha * normalized).to(destination.dtype)


class RecirculationMixAdapter:
    """Bind the non-model recurrence op to local tensors."""

    __slots__ = ("_config", "_destination_layer", "_source_layer")

    def __init__(
        self,
        config: RecirculationMixConfig,
        *,
        layer_count: int,
    ) -> None:
        source, destination = config.resolved_layers(layer_count)
        self._config = config
        self._source_layer = source
        self._destination_layer = destination

    def execute(
        self,
        op: RecurrenceOp,
        predecessors: tuple[tuple[RecurrenceOp, torch.Tensor], ...],
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if op.executor != "recirculation_mix" or op.point.pass_name != "mix":
            raise RuntimeError(
                f"native Recirculation cannot bind recurrence op {op.op_id!r}"
            )
        source = next(
            (
                value
                for predecessor, value in predecessors
                if predecessor.point.pass_name == "normal"
                and predecessor.point.layer == self._source_layer
            ),
            None,
        )
        destination = next(
            (
                value
                for predecessor, value in predecessors
                if predecessor.point.pass_name == "normal"
                and predecessor.point.layer == self._destination_layer
            ),
            None,
        )
        if source is None or destination is None:
            raise RuntimeError(
                "native Recirculation mix is missing source or destination state"
            )
        return mix_recirculation_tensors(
            self._config, source, destination, positions
        )


__all__ = ["RecirculationMixAdapter", "mix_recirculation_tensors"]
