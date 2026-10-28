"""Typed packed-row facts shared across backend adapters and runtime services."""

from __future__ import annotations

from dataclasses import dataclass

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.request_identity import RequestIdentity


@dataclass(frozen=True, slots=True)
class PackedPhaseRow:
    """One Rust-planned packed row with its stable request identity."""

    packed_row_index: int
    request_ordinal: int
    token_offset: int
    phase: HiddenPhase
    identity: RequestIdentity
    prompt_length: int = 0


@dataclass(frozen=True, slots=True)
class PackedPhaseRowBatch:
    """Immutable row expansion for one exact packed engine step."""

    plan_epoch: int
    step_id: int
    total_rows: int
    rows: tuple[PackedPhaseRow, ...]

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        return tuple(row.identity for row in self.rows)

    @property
    def request_rows(self) -> tuple[PackedPhaseRow, ...]:
        """Borrow each request's final original row in stable identity order."""
        last = {row.identity: row for row in self.rows}
        return tuple(
            sorted(
                last.values(),
                key=lambda row: (row.identity.request_id, row.identity.prompt_idx, row.identity.sample_idx),
            )
        )

    def producer_rows(self, phase: HiddenPhase | None) -> tuple[PackedPhaseRow, ...]:
        """Completed prompt endings or latest decode rows; None selects every request."""
        if phase is None:
            return self.request_rows
        return (
            tuple(row for row in self.prefill_rows if row.token_offset == row.prompt_length - 1)
            if phase is HiddenPhase.PREFILL
            else self.latest_decode_rows
        )

    @property
    def prefill_rows(self) -> tuple[PackedPhaseRow, ...]:
        return tuple(row for row in self.rows if row.phase is HiddenPhase.PREFILL)

    @property
    def decode_rows(self) -> tuple[PackedPhaseRow, ...]:
        return tuple(row for row in self.rows if row.phase is HiddenPhase.DECODE)

    @property
    def latest_decode_rows(self) -> tuple[PackedPhaseRow, ...]:
        """Each request's final decode row; earlier ones in its span recompute output history after preemption."""
        last = {row.request_ordinal: row for row in self.rows}
        return tuple(row for row in self.decode_rows if last[row.request_ordinal] is row)


@dataclass(frozen=True, slots=True)
class PackedSamplerRowLink:
    """One exact sampler-row to packed-model-row relation."""

    sampler_row_index: int
    packed_row_index: int


@dataclass(frozen=True, slots=True)
class PackedSamplerRowBatch:
    """Ordered sampler rows whose provenance is one exact packed row batch."""

    row_batch: PackedPhaseRowBatch
    links: tuple[PackedSamplerRowLink, ...]

    @property
    def plan_epoch(self) -> int:
        return self.row_batch.plan_epoch

    @property
    def step_id(self) -> int:
        return self.row_batch.step_id

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        return tuple(self.row_batch.rows[link.packed_row_index].identity for link in self.links)


__all__ = [
    "PackedPhaseRow",
    "PackedPhaseRowBatch",
    "PackedSamplerRowBatch",
    "PackedSamplerRowLink",
]
