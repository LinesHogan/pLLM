"""Phase-complete request identity staging for Rust-planned packed rows."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock

from tllm import _core
from tllm.contracts.hidden_locator import HiddenPhase
from tllm.contracts.packed_rows import (
    PackedPhaseRow,
    PackedPhaseRowBatch,
    PackedRowPlanInputs,
)
from tllm.contracts.request_identity import resolve_request_identity


class PackedPhaseRowStaging:
    """Stage exact packed-row identities before phase-specific tensor binding."""

    def __init__(self) -> None:
        self._batches: dict[tuple[int, int], PackedPhaseRowBatch] = {}
        self._plan_epoch: int | None = None
        self._retired_step = -1
        self._lock = RLock()

    def stage(
        self,
        plan_epoch: int,
        step_id: int,
        columns: PackedRowPlanInputs,
        *,
        prompt_indices: Mapping[str, int],
        sample_indices: Mapping[str, int],
    ) -> PackedPhaseRowBatch:
        """Expand one exact packed row plan into stable request identities."""

        if plan_epoch < 1:
            raise ValueError("packed phase staging plan epoch must be positive")
        if step_id < 0:
            raise ValueError("packed phase staging step id must be non-negative")
        key = (plan_epoch, step_id)
        with self._lock:
            if self._plan_epoch is not None:
                if plan_epoch < self._plan_epoch:
                    raise RuntimeError(
                        "packed phase staging cannot return to a prior plan epoch"
                    )
                if plan_epoch > self._plan_epoch and self._batches:
                    raise RuntimeError(
                        "packed phase staging cannot replace an active plan epoch"
                    )
                if (
                    plan_epoch == self._plan_epoch
                    and step_id <= self._retired_step
                ):
                    raise RuntimeError(
                        "packed phase staging step was already retired"
                    )
            row_plan = _core.plan_packed_prefill_rows(
                list(columns.scheduled_tokens),
                list(columns.prompt_tokens),
                list(columns.computed_tokens),
            )
            rows: list[PackedPhaseRow] = []
            decode_plan_ordinal = 0
            for request_ordinal, packed_row, token_offset, raw_phase in zip(
                row_plan.request_ordinals,
                row_plan.packed_row_indices,
                row_plan.token_offsets,
                row_plan.phases,
                strict=True,
            ):
                raw_request_id = columns.request_ids[request_ordinal]
                identity = resolve_request_identity(
                    raw_request_id,
                    prompt_indices=prompt_indices,
                    sample_indices=sample_indices,
                )
                if identity is None:
                    raise RuntimeError(
                        "request-state packed row has no stable request identity"
                    )
                phase = HiddenPhase(raw_phase)
                ordinal: int | None = None
                if phase is HiddenPhase.DECODE:
                    ordinal = decode_plan_ordinal
                    decode_plan_ordinal += 1
                rows.append(
                    PackedPhaseRow(
                        packed_row_index=packed_row,
                        request_ordinal=request_ordinal,
                        raw_request_id=raw_request_id,
                        token_offset=token_offset,
                        phase=phase,
                        identity=identity,
                        decode_plan_ordinal=ordinal,
                    )
                )
            batch = PackedPhaseRowBatch(
                plan_epoch=plan_epoch,
                step_id=step_id,
                total_rows=row_plan.total_rows,
                rows=tuple(rows),
            )
            previous = self._batches.get(key)
            if previous is not None and previous != batch:
                raise RuntimeError(
                    "packed phase staging differs from its staged step"
                )
            if self._plan_epoch != plan_epoch:
                self._plan_epoch = plan_epoch
                self._retired_step = -1
            self._batches[key] = batch
        return batch

    def batch_for_step(
        self, plan_epoch: int, step_id: int
    ) -> PackedPhaseRowBatch:
        with self._lock:
            try:
                return self._batches[(plan_epoch, step_id)]
            except KeyError as exc:
                raise RuntimeError(
                    "packed phase rows were not staged for this step"
                ) from exc

    def finish_step(self, plan_epoch: int, step_id: int) -> None:
        with self._lock:
            if self._batches.pop((plan_epoch, step_id), None) is None:
                raise RuntimeError(
                    "packed phase rows were not staged for this step"
                )
            self._retired_step = max(self._retired_step, step_id)


__all__ = ["PackedPhaseRowStaging"]
