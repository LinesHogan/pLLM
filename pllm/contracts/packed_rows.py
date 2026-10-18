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

    def __post_init__(self) -> None:
        if self.packed_row_index < 0 or self.request_ordinal < 0 or self.token_offset < 0:
            raise ValueError("packed phase row indices must be non-negative")
        if not self.identity.request_id or self.identity.prompt_idx < 0 or self.identity.sample_idx < 0:
            raise ValueError("packed phase row requires a resolved request identity")
        if self.phase not in (HiddenPhase.PREFILL, HiddenPhase.DECODE):
            raise ValueError("packed row requires prefill or decode phase")


@dataclass(frozen=True, slots=True)
class PackedPhaseRowBatch:
    """Immutable row expansion for one exact packed engine step."""

    plan_epoch: int
    step_id: int
    total_rows: int
    rows: tuple[PackedPhaseRow, ...]

    def __post_init__(self) -> None:
        if self.plan_epoch < 1:
            raise ValueError("packed phase batch plan epoch must be positive")
        if self.step_id < 0 or self.total_rows < 0:
            raise ValueError("packed phase batch facts must be non-negative")
        if self.total_rows != len(self.rows):
            raise ValueError("packed phase batch must contain every planned row")
        if tuple(row.packed_row_index for row in self.rows) != tuple(range(self.total_rows)):
            raise ValueError("packed phase batch rows must preserve packed plan order")

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

    @property
    def prefill_rows(self) -> tuple[PackedPhaseRow, ...]:
        return tuple(row for row in self.rows if row.phase is HiddenPhase.PREFILL)

    @property
    def decode_rows(self) -> tuple[PackedPhaseRow, ...]:
        return tuple(row for row in self.rows if row.phase is HiddenPhase.DECODE)


@dataclass(frozen=True, slots=True)
class PackedSamplerRowLink:
    """One exact sampler-row to packed-model-row relation."""

    sampler_row_index: int
    packed_row_index: int

    def __post_init__(self) -> None:
        if self.sampler_row_index < 0 or self.packed_row_index < 0:
            raise ValueError("packed sampler row indices must be non-negative")


@dataclass(frozen=True, slots=True)
class PackedSamplerRowBatch:
    """Ordered sampler rows whose provenance is one exact packed row batch."""

    row_batch: PackedPhaseRowBatch
    links: tuple[PackedSamplerRowLink, ...]

    def __post_init__(self) -> None:
        if tuple(link.sampler_row_index for link in self.links) != tuple(range(len(self.links))):
            raise ValueError("packed sampler links must preserve sampler row order")
        if any(link.packed_row_index >= self.row_batch.total_rows for link in self.links):
            raise ValueError("packed sampler link is outside its packed row batch")

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
