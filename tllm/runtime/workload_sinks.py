"""Runtime-owned storage for signed workload sink publications."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.runtime.decode_runtime_metadata import active_request_prompt_sample_metadata


@dataclass(frozen=True, slots=True)
class StepRowIdentity:
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
    value: torch.Tensor
    identity: StepRowIdentity


@dataclass(frozen=True, slots=True)
class MissingSinkRecord:
    pass


SinkLookup = SinkRecord | MissingSinkRecord


class SinkStore:
    """Keep one runtime-validated publication per signed sink."""

    def __init__(self) -> None:
        self._records: dict[str, SinkRecord] = {}

    def publish(self, sink: str, value: torch.Tensor, step_id: int) -> None:
        if value.ndim != 2:
            raise ValueError(f"sink {sink!r} requires a matrix payload")
        from tllm.runtime import residual_runtime

        request_ids, prompt_idxs, sample_idxs = active_request_prompt_sample_metadata(
            residual_runtime.RUNTIME, value.shape[0]
        )
        self._records[sink] = SinkRecord(
            value=value.detach().clone(),
            identity=StepRowIdentity(
                request_ids=tuple(request_ids),
                prompt_idxs=tuple(prompt_idxs),
                sample_idxs=tuple(sample_idxs),
                phase="decode",
                engine_step_id=step_id,
            ),
        )

    def lookup(self, sink: str, step_id: int) -> SinkLookup:
        record = self._records.get(sink)
        if record is None or record.identity.engine_step_id != step_id:
            return MissingSinkRecord()
        return record

    def latest_rows(self, sink: str) -> int:
        record = self._records.get(sink)
        match record:
            case None:
                return 0
            case SinkRecord(identity=identity):
                return identity.rows


__all__ = [
    "MissingSinkRecord",
    "SinkLookup",
    "SinkRecord",
    "SinkStore",
    "StepRowIdentity",
]
