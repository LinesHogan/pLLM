"""Mechanical transport of vLLM scheduler columns into the Rust phase enum."""

from __future__ import annotations

from tllm import _core
from tllm.contracts.packed_rows import PackedRowPlanInputs


def scheduler_batch_phase(
    *,
    columns: PackedRowPlanInputs,
) -> _core.SchedulerBatchPhase:
    """Classify columns already adapted by the bound MRv1/MRv2 driver."""
    try:
        return _core.classify_scheduler_batch_phase(
            list(columns.scheduled_tokens),
            list(columns.prompt_tokens),
            list(columns.computed_tokens),
        )
    except ValueError as exc:
        raise RuntimeError(
            "Rust scheduler batch phase rejected vLLM scheduler columns: "
            f"{exc}"
        ) from exc
