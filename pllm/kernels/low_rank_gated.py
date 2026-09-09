"""Batched functional low-rank gated residual tensor math."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pllm.kernels.paged_linear import PagedRows, TensorRows


@dataclass(frozen=True, slots=True)
class LowRankGatedLayout:
    hidden_size: int
    rank: int
    output_layernorm: bool

    def __post_init__(self) -> None:
        if (
            self.hidden_size < 1
            or self.rank < 1
            or self.rank > self.hidden_size
        ):
            raise ValueError("low-rank gated layout dimensions are invalid")

    @property
    def parameter_count(self) -> int:
        projection_parameters = 3 * self.hidden_size * self.rank + self.rank
        if self.output_layernorm:
            return projection_parameters + 2 * self.hidden_size
        return projection_parameters


def low_rank_gated_forward(
    source: torch.Tensor,
    rows: TensorRows | PagedRows,
    layout: LowRankGatedLayout,
) -> torch.Tensor:
    """Evaluate packed rows against explicitly indexed flat state vectors."""

    output, _, _ = _residual(source, rows, layout)
    if layout.output_layernorm:
        start = 3 * layout.hidden_size * layout.rank + layout.rank
        output = F.layer_norm(output, (layout.hidden_size,)) * rows.values(start, layout.hidden_size)
        output = output + rows.values(start + layout.hidden_size, layout.hidden_size)
    return output


def _residual(source: torch.Tensor, pages: PagedRows | TensorRows,
                    layout: LowRankGatedLayout) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden, rank = layout.hidden_size, layout.rank
    if (source.ndim != 2 or source.shape[1] != hidden or pages.pool.shape[1] != layout.parameter_count
        or source.device != pages.pool.device or source.dtype != pages.pool.dtype):
        raise ValueError("low-rank gated tensors differ from their frozen layout")
    up = pages.linear(source, 0, (hidden, rank))
    gate = pages.linear(source, hidden * rank, (hidden, rank)) + pages.values(3 * hidden * rank, rank)
    return source + pages.linear(up * F.silu(gate), 2 * hidden * rank, (rank, hidden)), up, gate


@torch.no_grad()
def low_rank_gated_loss_and_gradients(source: torch.Tensor, target: torch.Tensor,
                                     pages: TensorRows | PagedRows,
                                     layout: LowRankGatedLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Negative row indices are inactive padding in a cold captured batch envelope."""
    hidden, rank = layout.hidden_size, layout.rank
    width, norm_start = hidden * rank, 3 * hidden * rank + rank
    raw, up, gate = _residual(source, pages, layout)
    counts = pages.reduce(source.new_ones((len(source),))).clamp_min(1)
    normalization = pages.broadcast(counts).unsqueeze(1) * hidden
    extras: tuple[torch.Tensor, ...] = ()
    gradient: torch.Tensor
    if layout.output_layernorm:
        scale = pages.values(norm_start, hidden)
        normalized, mean, rstd = torch.native_layer_norm(raw, (hidden,), None, None, 1e-5)
        output = normalized * scale + pages.values(norm_start + hidden, hidden)
        dy = 2 * (output - target) / normalization
        extras = (pages.reduce(dy * normalized), pages.reduce(dy))
        gradient, _, _ = torch.ops.aten.native_layer_norm_backward(dy * scale, raw, (hidden,),
            mean, rstd, None, None, (True, False, False))
    else:
        output = raw
        gradient = 2 * (output - target) / normalization
    losses = pages.reduce((output - target).square().mean(1)) / counts
    activation = F.silu(gate)
    product_gradient = pages.input_gradient(gradient, 2 * width, (rank, hidden))
    up_gradient = product_gradient * activation
    sigmoid = torch.sigmoid(gate)
    gate_gradient = product_gradient * up * sigmoid * (1 + gate * (1 - sigmoid))
    gradients = (pages.parameter_gradient(source, up_gradient).flatten(1),
        pages.parameter_gradient(source, gate_gradient).flatten(1),
        pages.parameter_gradient(up * activation, gradient).flatten(1), pages.reduce(gate_gradient), *extras)
    return losses, torch.cat(gradients, dim=1)
