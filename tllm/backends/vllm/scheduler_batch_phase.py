"""Mechanical transport of vLLM scheduler columns into the Rust phase enum."""

from __future__ import annotations

from tllm import _core
from tllm.backends.vllm.runner_adapter import PackedRowPlanInputs
from tllm.runtime.step_session import StepSession


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
