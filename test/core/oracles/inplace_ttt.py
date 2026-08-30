"""Frozen exact fallback oracle from the pre-native fast-weight implementation.

Provenance: ``git show 12d2f2b^:tllm/runtime/fast_weight/state.py``. This is
the repository's historical exact update rule, not an upstream checkpoint
oracle with ``ttt_proj``/``ttt_conv`` parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class OracleInPlaceTTTResult:
    output: torch.Tensor
    tail_hidden: torch.Tensor
    tail_target: torch.Tensor
    weight: torch.Tensor
    update_count: int
    trained_rows: int


def oracle_exact_step(
    hidden: torch.Tensor,
    target: torch.Tensor,
    base_weight: torch.Tensor,
    *,
    chunk_size: int,
    learning_rate: float,
    past_hidden: torch.Tensor | None = None,
    past_target: torch.Tensor | None = None,
    past_weight: torch.Tensor | None = None,
) -> OracleInPlaceTTTResult:
    work_hidden = (
        hidden
        if past_hidden is None
        else torch.cat((past_hidden, hidden), dim=0)
    )
    work_target = (
        target
        if past_target is None
        else torch.cat((past_target, target), dim=0)
    )
    weight = (
        base_weight.clone()
        if past_weight is None
        else past_weight.clone()
    )
    full_rows = (work_hidden.shape[0] // chunk_size) * chunk_size
    outputs: list[torch.Tensor] = []
    updates = 0
    for start in range(0, full_rows, chunk_size):
        chunk_hidden = work_hidden[start : start + chunk_size]
        chunk_target = work_target[start : start + chunk_size]
        outputs.append(chunk_hidden @ weight.transpose(0, 1))
        weight = weight + learning_rate * (
            chunk_target.transpose(0, 1) @ chunk_hidden
        ).to(weight)
        updates += 1
    tail_hidden = work_hidden[full_rows:]
    tail_target = work_target[full_rows:]
    if tail_hidden.shape[0]:
        outputs.append(tail_hidden @ weight.transpose(0, 1))
    output_all = torch.cat(outputs, dim=0)
    return OracleInPlaceTTTResult(
        output=output_all[-hidden.shape[0] :],
        tail_hidden=tail_hidden.clone(),
        tail_target=tail_target.clone(),
        weight=weight,
        update_count=updates,
        trained_rows=full_rows,
    )


__all__ = ["OracleInPlaceTTTResult", "oracle_exact_step"]
