"""Batched functional low-rank gated residual tensor math."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class LowRankGatedLayout:
    hidden_size: int
    rank: int
    output_layernorm: bool

    def __post_init__(self) -> None:
        if (
            type(self.hidden_size) is not int
            or type(self.rank) is not int
            or self.hidden_size < 1
            or self.rank < 1
            or self.rank > self.hidden_size
            or type(self.output_layernorm) is not bool
        ):
            raise ValueError("low-rank gated layout dimensions are invalid")

    @property
    def parameter_count(self) -> int:
        projection_parameters = 3 * self.hidden_size * self.rank + self.rank
        if self.output_layernorm:
            return projection_parameters + 2 * self.hidden_size
        return projection_parameters


def initialize_low_rank_gated(
    layout: LowRankGatedLayout,
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create one random flat parameter image for a cold PTTS admission."""

    hidden_rank = layout.hidden_size * layout.rank
    a_end = hidden_rank
    g_end = 2 * hidden_rank
    b_end = 3 * hidden_rank
    gate_bias_end = b_end + layout.rank
    parameters = torch.empty(
        (layout.parameter_count,),
        dtype=dtype,
        device=device,
    )
    with torch.no_grad():
        parameters[:a_end].normal_(mean=0.0, std=0.02, generator=generator)
        parameters[a_end:g_end].normal_(
            mean=0.0,
            std=0.02,
            generator=generator,
        )
        parameters[g_end:b_end].normal_(
            mean=0.0,
            std=0.02,
            generator=generator,
        )
        parameters[b_end:gate_bias_end].fill_(1.0)
        if layout.output_layernorm:
            output_weight_end = gate_bias_end + layout.hidden_size
            parameters[gate_bias_end:output_weight_end].fill_(1.0)
            parameters[output_weight_end:].zero_()
    return parameters


def low_rank_gated_forward(
    source: torch.Tensor,
    state_vectors: torch.Tensor,
    row_state_indices: torch.Tensor,
    layout: LowRankGatedLayout,
) -> torch.Tensor:
    """Evaluate packed rows against explicitly indexed flat state vectors."""

    if (
        source.ndim != 2
        or state_vectors.ndim != 2
        or row_state_indices.ndim != 1
        or int(source.shape[0]) != int(row_state_indices.shape[0])
        or int(source.shape[1]) != layout.hidden_size
        or int(state_vectors.shape[1]) != layout.parameter_count
        or row_state_indices.dtype is not torch.long
        or source.dtype is not state_vectors.dtype
        or source.device != state_vectors.device
        or source.device != row_state_indices.device
    ):
        raise ValueError("low-rank gated tensors differ from their frozen layout")
    hidden_rank = layout.hidden_size * layout.rank
    a_end = hidden_rank
    g_end = 2 * hidden_rank
    b_end = 3 * hidden_rank
    gate_bias_end = b_end + layout.rank
    state_count = int(state_vectors.shape[0])
    a = state_vectors[:, :a_end].reshape(
        state_count,
        layout.hidden_size,
        layout.rank,
    )
    g = state_vectors[:, a_end:g_end].reshape(
        state_count,
        layout.hidden_size,
        layout.rank,
    )
    b = state_vectors[:, g_end:b_end].reshape(
        state_count,
        layout.rank,
        layout.hidden_size,
    )
    gate_bias = state_vectors[:, b_end:gate_bias_end]
    selected_a = a.index_select(0, row_state_indices)
    selected_g = g.index_select(0, row_state_indices)
    selected_b = b.index_select(0, row_state_indices)
    selected_bias = gate_bias.index_select(0, row_state_indices)
    source_rows = source.unsqueeze(1)
    up = torch.bmm(source_rows, selected_a).squeeze(1)
    gate = F.silu(torch.bmm(source_rows, selected_g).squeeze(1) + selected_bias)
    output = torch.bmm((up * gate).unsqueeze(1), selected_b).squeeze(1) + source
    if not layout.output_layernorm:
        return output
    output_weight_end = gate_bias_end + layout.hidden_size
    output_weight = state_vectors[
        :, gate_bias_end:output_weight_end
    ].index_select(0, row_state_indices)
    output_bias = state_vectors[:, output_weight_end:].index_select(
        0,
        row_state_indices,
    )
    normalized = F.layer_norm(output, (layout.hidden_size,))
    return normalized * output_weight + output_bias
