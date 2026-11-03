"""Typed packed-row facts shared across backend adapters and runtime services."""

from __future__ import annotations

from dataclasses import dataclass

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.request_identity import RequestIdentity


@dataclass(frozen=True, slots=True)
class PackedPhaseRow:
    """One packed row with its stable request identity."""

    packed_row_index: int
    request_ordinal: int
    token_offset: int
    phase: HiddenPhase
    identity: RequestIdentity
    prompt_length: int = 0
    # Retained auxiliary state has already consumed this token; reset-and-replay leaves it false.
    recomputed: bool = False


@dataclass(frozen=True, slots=True)
class PackedPhaseRowBatch:
    """Immutable row expansion for one exact packed engine step."""

    step_id: int
    total_rows: int
    rows: tuple[PackedPhaseRow, ...]

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        return tuple(row.identity for row in self.rows)

    @property
    def request_rows(self) -> tuple[PackedPhaseRow, ...]:
        """Borrow each request's final row in packed request order, shared by producers and consumers."""
        last = {row.identity: row for row in self.rows}
        return tuple(last.values())

    def producer_rows(self, phase: HiddenPhase | None) -> tuple[PackedPhaseRow, ...]:
        """Fresh prompt endings or latest decode rows; retained-state replay never produces again."""
        if phase is None:
            return tuple(row for row in self.request_rows if not row.recomputed)
        return (
            tuple(row for row in self.prefill_rows if row.token_offset == row.prompt_length - 1 and not row.recomputed)
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
        """Each request's final fresh decode row; a partial retained-state replay has none."""
        last = {row.request_ordinal: row for row in self.rows}
        return tuple(
            row
            for row in self.rows
            if row.phase is HiddenPhase.DECODE and last[row.request_ordinal] is row and not row.recomputed
        )


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
    def step_id(self) -> int:
        return self.row_batch.step_id

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        return tuple(self.row_batch.rows[link.packed_row_index].identity for link in self.links)


def label_rows(
    identities: tuple[RequestIdentity, ...],
    scheduled: list[int],
    prompt: list[int],
    computed: list[int],
    step_id: int,
) -> PackedSamplerRowBatch:
    """One row per scheduled token in request order; token positions below the prompt length are prefill."""
    rows: list[PackedPhaseRow] = []
    links: list[PackedSamplerRowLink] = []
    for request, (count, length, done) in enumerate(zip(scheduled, prompt, computed, strict=True)):
        for offset in range(done, done + count):
            phase = HiddenPhase.PREFILL if offset < length else HiddenPhase.DECODE
            rows.append(PackedPhaseRow(len(rows), request, offset, phase, identities[request], length))
        # vLLM samples each request's last scheduled token, which is the last packed row of its span.
        links.append(PackedSamplerRowLink(request, len(rows) - 1))
    return PackedSamplerRowBatch(PackedPhaseRowBatch(step_id, len(rows), tuple(rows)), tuple(links))


__all__ = [
    "label_rows",
    "PackedPhaseRow",
    "PackedPhaseRowBatch",
    "PackedSamplerRowBatch",
    "PackedSamplerRowLink",
]
