"""Mechanical transport of vLLM scheduler columns into the Rust phase enum."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from tllm import _core
from tllm.contracts.packed_rows import PackedPhaseRowBatch, PackedRowPlanInputs
from tllm.backends.vllm._step_session import StepSession


class PreparedPackedRowsExecution(Protocol):
    """Native execution surface required at the vLLM prepare-input boundary."""

    @property
    def requires_packed_rows(self) -> bool: ...

    def stage_packed_row_plan(
        self,
        step_id: int,
        columns: PackedRowPlanInputs,
        prompt_indices: Mapping[str, int],
        sample_indices: Mapping[str, int],
    ) -> PackedPhaseRowBatch: ...


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


def hook_phase(phase: _core.SchedulerBatchPhase) -> str:
    """Mechanically transport a verified Rust enum to the hook boundary.

    ``mixed`` is deliberately not collapsed to either row phase.  Native
    Region advancement is keyed by the signed backend event, while the
    compatibility dispatcher rejects mixed batches when a legacy event
    subscriber would otherwise need to interpret the aggregate batch.
    """
    if phase == _core.SchedulerBatchPhase.Prefill:
        return "prefill"
    if phase == _core.SchedulerBatchPhase.Decode:
        return "decode"
    if phase == _core.SchedulerBatchPhase.Empty:
        return "empty"
    if phase == _core.SchedulerBatchPhase.Mixed:
        return "mixed"
    raise RuntimeError("vLLM hook lifecycle received an unknown Rust batch phase")


def session_hook_phase(session: StepSession | None) -> str:
    """Require the phase signed by Rust at the input boundary for this step."""
    if session is None or session.batch_phase is None:
        raise RuntimeError("vLLM step session has no Rust-derived scheduler batch phase")
    return hook_phase(session.batch_phase)


def stage_prepared_packed_rows(
    *,
    execution: PreparedPackedRowsExecution | None,
    step_id: int,
    columns: PackedRowPlanInputs,
    prompt_indices: Mapping[str, int],
    sample_indices: Mapping[str, int],
) -> PackedPhaseRowBatch | None:
    """Stage exact packed rows when the prepared native plan consumes them.

    Driver wrappers provide only their normalized scheduler columns and the
    backend-resolved request mappings.  Rust expands phase rows, and the
    native execution validates that the resulting batch belongs to its
    already-prepared plan epoch and step.
    """

    if execution is None or not execution.requires_packed_rows:
        return None
    return execution.stage_packed_row_plan(
        step_id,
        columns,
        prompt_indices,
        sample_indices,
    )
