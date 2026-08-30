"""Concrete signed cleanup service for one native request-finish batch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tllm.contracts.request_identity import RequestFinishedBatch

if TYPE_CHECKING:
    from tllm.backends.protocol import RunnerBinding
    from tllm.runtime.state import ResidualRuntimeState


@dataclass(slots=True)
class NativeRequestFinishCleanup:
    """Run a lifecycle-issued finish batch through one local native runtime.

    This service intentionally has no knowledge of raw request aliases or
    ledger ownership.  It only stages/runs signed work and lets the optional
    overlay adapter consume the already-canonical batch afterwards.
    """

    runtime: "ResidualRuntimeState"
    binding: "RunnerBinding | None"

    def stage(self, batch: RequestFinishedBatch) -> int:
        execution = self.runtime.workload_execution
        return 0 if execution is None else execution.stage_request_finished(batch)

    def confirm(self, batch: RequestFinishedBatch, staged_ports: int) -> None:
        execution = self.runtime.workload_execution
        if staged_ports:
            step = int(self.runtime.event_step_id) + 1
            if execution is None:
                raise RuntimeError("request-finish flush has no active native execution")
            execution.execute_staged_request_finished(step)
            self.runtime.event_step_id = step
        if execution is not None:
            # Native request-state retirement drains async host work and
            # verifies all typed state-effect receipts at the safe point.
            if not staged_ports:
                execution.finish_step(int(self.runtime.event_step_id))
            execution.retire_request_states(batch)
        binding = self.binding
        if binding is not None:
            binding.driver.consume_parameter_overlay_finished_batch(
                runtime=self.runtime,
                runner=binding.runner,
                state=binding.parameter_overlay_state,
                batch=batch,
            )
            binding.driver.synchronize_parameter_overlays(
                runtime=self.runtime,
                runner=binding.runner,
                state=binding.parameter_overlay_state,
            )


__all__ = ["NativeRequestFinishCleanup"]
