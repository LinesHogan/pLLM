"""Framework-managed sink publication storage.

A ``sink("name")`` task output is published by the engine at a signed point
in the step; the row identity of that publication (which requests/prompts/
samples the rows belong to, at which engine step) is a fact the *runtime*
already holds — authors must never assemble or carry it.  ``SinkStore``
records each publication with the identity captured from the runtime at
publish time, and answers step-scoped lookups for the sampler bridge.

One record per sink name: a publication belongs to its step, and the signed
timing guarantees consumption within that step (the sampler bridge joins the
publication fence before reading), so retention is a single latest record.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.runtime.decode_runtime_metadata import active_request_prompt_sample_metadata


@dataclass(frozen=True, slots=True)
class StepRowIdentity:
    """Which engine rows a sink publication covers, captured by the runtime."""

    request_ids: tuple[str, ...]
    prompt_idxs: tuple[int, ...]
    sample_idxs: tuple[int, ...]
    phase: str
    engine_step_id: int

    @property
    def rows(self) -> int:
        return len(self.request_ids)


@dataclass(frozen=True, slots=True)
class SinkRecord:
    """One sink publication: the payload plus its runtime-captured identity."""

    value: torch.Tensor
    identity: StepRowIdentity


class SinkStore:
    """Latest publication per sink name, with runtime-attached row identity."""

    def __init__(self) -> None:
        self._records: dict[str, SinkRecord] = {}

    def publish(self, sink: str, value: object, step_id: int) -> None:
        """Record one signed sink publication; called by the generated binding."""
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise TypeError(f"sink {sink!r} publications must be rank-2 Tensors")
        from tllm.runtime import residual_runtime

        rows = int(value.shape[0])
        request_ids, prompt_idxs, sample_idxs = active_request_prompt_sample_metadata(
            residual_runtime.RUNTIME, rows
        )
        self._records[sink] = SinkRecord(
            value=value.detach().clone(),
            identity=StepRowIdentity(
                request_ids=tuple(request_ids),
                prompt_idxs=tuple(prompt_idxs),
                sample_idxs=tuple(sample_idxs),
                # Sink outputs exist only on the decode phase today; the
                # declared hidden() inputs pin the phase in the signed IR.
                phase="decode",
                engine_step_id=int(step_id),
            ),
        )

    def for_step(self, sink: str, step_id: int) -> SinkRecord | None:
        """Latest record for ``sink`` if it belongs to ``step_id`` (kit-internal)."""
        record = self._records.get(sink)
        if record is None or record.identity.engine_step_id != int(step_id):
            return None
        return record

    def latest_rows(self, sink: str) -> int:
        """Row count of the latest publication on ``sink`` (0 when none)."""
        record = self._records.get(sink)
        return 0 if record is None else record.identity.rows


__all__ = ["SinkRecord", "SinkStore", "StepRowIdentity"]
