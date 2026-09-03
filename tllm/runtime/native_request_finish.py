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
        return 0

    def confirm(self, batch: RequestFinishedBatch, staged_ports: int) -> None:
        if staged_ports:
            raise RuntimeError("native request-finish execution is retired")
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
