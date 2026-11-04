"""Preemption state policy shared by engine sessions and schedule diagnostics."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from pllm.contracts.backend_attachment import BackendAttachment
from pllm.contracts.packed_rows import PackedSamplerRowBatch
from pllm.contracts.request_identity import RequestIdentity

if TYPE_CHECKING:
    from pllm.execution.model_sites import ResidualRewrite


def preemption_policy(attachment: BackendAttachment, sites: tuple[ResidualRewrite, ...]) -> Literal["keep", "reset"]:
    """Model-state effects replay from admission; sampler and hidden-only effects retain state.

    A reset also needs the plan's recompute_rejection to be empty before replay begins.
    """
    adaptive = any(
        task.custom_forward is not None and task.custom_forward.adaptive
        for task in attachment.plan.semantic_graph.tasks
    )
    return (
        "reset"
        if attachment.spec.overlay_bindings
        or attachment.spec.activation_bindings
        or sites
        or attachment.probe.custom_attention
        or adaptive
        else "keep"
    )


def retained_rows(rows: PackedSamplerRowBatch, frontiers: dict[RequestIdentity, int]) -> PackedSamplerRowBatch:
    """Mark previously consumed tokens when state survives replay, including partial replay chunks.

    The session and CPU checker own the frontier map and remove entries on retirement.
    Reset-and-replay workloads do not call this: their model-state transitions must run again.
    """
    batch = rows.row_batch
    labeled = tuple(
        replace(row, recomputed=True) if row.token_offset < frontiers.get(row.identity, 0) else row
        for row in batch.rows
    )
    for row in batch.request_rows:
        frontiers[row.identity] = max(frontiers.get(row.identity, 0), row.token_offset + 1)
    return replace(rows, row_batch=replace(batch, rows=labeled))
