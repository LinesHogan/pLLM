"""Runtime-owned storage for signed workload sink publications."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.ports.sampler import (
    SamplerHiddenReweightBatch,
    SamplerOutputValue,
)
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
    value: SamplerOutputValue
    identity: StepRowIdentity


@dataclass(frozen=True, slots=True)
class MissingSinkRecord:
    pass


SinkLookup = SinkRecord | MissingSinkRecord


class SinkStore:
    """Keep one runtime-validated publication per signed sink."""

    def __init__(self) -> None:
        self._records: dict[str, SinkRecord] = {}

    def publish(
        self, sink: str, value: SamplerOutputValue, step_id: int
    ) -> None:
        from tllm.runtime import residual_runtime

        match value:
            case SamplerHiddenReweightBatch(
                pred_hidden=pred_hidden,
                affected_row_ids=affected_row_ids,
                total_rows=total_rows,
            ):
                if pred_hidden.ndim != 2:
                    raise ValueError(
                        f"sink {sink!r} requires rank-2 hidden predictions"
                    )
                if int(pred_hidden.shape[0]) != len(affected_row_ids):
                    raise ValueError(
                        f"sink {sink!r} hidden predictions must align with affected rows"
                    )
                if total_rows < 0:
                    raise ValueError(
                        f"sink {sink!r} total rows must be non-negative"
                    )
                if affected_row_ids != tuple(sorted(set(affected_row_ids))):
                    raise ValueError(
                        f"sink {sink!r} affected rows must be sorted and unique"
                    )
                if any(
                    row_id < 0 or row_id >= total_rows
                    for row_id in affected_row_ids
                ):
                    raise ValueError(
                        f"sink {sink!r} affected rows are out of bounds"
                    )
                stored_value: SamplerOutputValue = SamplerHiddenReweightBatch(
                    pred_hidden.detach().clone(),
                    affected_row_ids,
                    total_rows,
                )
                active_rows = total_rows
            case torch.Tensor() as tensor:
                if tensor.ndim != 2:
                    raise ValueError(
                        f"sink {sink!r} requires a matrix payload"
                    )
                stored_value = tensor.detach().clone()
                active_rows = int(tensor.shape[0])
        request_ids, prompt_idxs, sample_idxs = active_request_prompt_sample_metadata(
            residual_runtime.RUNTIME, active_rows
        )
        self._records[sink] = SinkRecord(
            value=stored_value,
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
            case SinkRecord(
                value=SamplerHiddenReweightBatch(
                    affected_row_ids=affected_row_ids
                )
            ):
                return len(affected_row_ids)
            case SinkRecord(identity=identity):
                return identity.rows


__all__ = [
    "MissingSinkRecord",
    "SinkLookup",
    "SinkRecord",
    "SinkStore",
    "StepRowIdentity",
]
