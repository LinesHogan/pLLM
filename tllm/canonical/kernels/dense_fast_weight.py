"""Pure dense fast-weight transitions over one fixed-shape state root."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DenseFastWeightTransition:
    """The next immutable root fields and outputs for one dense transition."""

    output: torch.Tensor
    delta_weight: torch.Tensor
    tail_input: torch.Tensor
    tail_target: torch.Tensor
    tail_length: int
    update_count: int
    trained_rows: int


def transition_dense_fast_weight(
    *,
    base_weight: torch.Tensor,
    delta_weight: torch.Tensor,
    tail_input: torch.Tensor,
    tail_target: torch.Tensor,
    tail_length: int,
    hidden: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
    learning_rate: float,
) -> DenseFastWeightTransition:
    """Advance a delta-weight root while retaining its incomplete input pairs.

    Each complete chunk emits with its pre-update effective weight and then
    adds ``learning_rate * target.T @ hidden`` to the delta.  The fixed tail
    buffers retain the unconsumed suffix for the next transition.
    """

    working_input = torch.cat((tail_input[:tail_length], hidden), dim=0)
    working_target = torch.cat((tail_target[:tail_length], target), dim=0)
    complete_rows = (working_input.shape[0] // chunk_size) * chunk_size
    next_delta = delta_weight.clone()
    outputs: list[torch.Tensor] = []
    update_count = 0

    for start in range(0, complete_rows, chunk_size):
        chunk_input = working_input[start : start + chunk_size]
        chunk_target = working_target[start : start + chunk_size]
        outputs.append(chunk_input @ (base_weight + next_delta).transpose(0, 1))
        next_delta = next_delta + (
            learning_rate * (chunk_target.transpose(0, 1) @ chunk_input)
        ).to(next_delta)
        update_count += 1

    next_tail_length = working_input.shape[0] - complete_rows
    next_tail_input = torch.zeros_like(tail_input)
    next_tail_target = torch.zeros_like(tail_target)
    next_tail_input[:next_tail_length].copy_(working_input[complete_rows:])
    next_tail_target[:next_tail_length].copy_(working_target[complete_rows:])
    if next_tail_length:
        outputs.append(
            working_input[complete_rows:]
            @ (base_weight + next_delta).transpose(0, 1)
        )

    return DenseFastWeightTransition(
        output=torch.cat(outputs, dim=0)[-hidden.shape[0] :],
        delta_weight=next_delta,
        tail_input=next_tail_input,
        tail_target=next_tail_target,
        tail_length=next_tail_length,
        update_count=update_count,
        trained_rows=complete_rows,
    )


__all__ = ["DenseFastWeightTransition", "transition_dense_fast_weight"]
