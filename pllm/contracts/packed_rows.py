"""Typed packed-row facts shared across backend adapters and runtime services."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import overload

import numpy as np
import numpy.typing as npt

from pllm import _core
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.request_identity import RequestIdentity

Slots = npt.NDArray[np.int64]


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

    @property
    def role(self) -> HiddenPhase:
        """Prompt or generated token, also when generated history arrives in a prefill phase."""
        return HiddenPhase.DECODE if self.token_offset >= self.prompt_length else HiddenPhase.PREFILL


def identity_words(identity: RequestIdentity) -> Slots:
    """Encode an admitted identity once for native projection, without a process-global interner."""
    request = identity.request_id.encode()
    return np.array(
        (identity.prompt_index, identity.sample_index, len(request), *request, *identity.route.encode()), dtype=np.int64
    )


@dataclass(frozen=True, slots=True, eq=False)
class PackedRows(Sequence[PackedPhaseRow]):
    """Numeric selection of packed rows; individual records are created only when inspected."""

    batch: PackedPhaseRowBatch
    indices: Slots

    @classmethod
    def from_rows(cls, rows: Sequence[PackedPhaseRow]) -> PackedRows:
        """Import explicit records at a diagnostic or shape-tracing boundary."""
        return PackedPhaseRowBatch.from_rows(0, len(rows), tuple(rows)).rows

    def __len__(self) -> int:
        return len(self.indices)

    @overload
    def __getitem__(self, index: int) -> PackedPhaseRow: ...

    @overload
    def __getitem__(self, index: slice) -> PackedRows: ...

    def __getitem__(self, index: int | slice) -> PackedPhaseRow | PackedRows:
        return (
            PackedRows(self.batch, np.ascontiguousarray(self.indices[index]))
            if isinstance(index, slice)
            else self.batch.inspect_row(int(self.indices[index]))
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PackedRows):
            if self.batch is other.batch:
                return np.array_equal(self.indices, other.indices)
            return (
                np.array_equal(self.columns[[0, 1, 2, 3, 5, 6]], other.columns[[0, 1, 2, 3, 5, 6]])
                and self.identities == other.identities
            )
        return tuple(self) == other

    @property
    def columns(self) -> Slots:
        """Borrow selected numeric labels in invocation order."""
        return self.batch.columns[:, self.indices]

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        """Borrow admitted identities in row order without constructing row records."""
        identities = np.asarray(self.batch.request_identities, dtype=np.object_)
        return tuple(identities[self.batch.columns[4, self.indices]])

    def producer(self, phase: HiddenPhase | None) -> PackedRows:
        """Select a producer phase without inspecting records."""
        if phase is None:
            return self
        selected = self.batch.tap_selected[int(phase is HiddenPhase.PREFILL)]
        return PackedRows(self.batch, self.indices[np.isin(self.indices, selected)])


@dataclass(frozen=True, slots=True, eq=False)
class PackedPhaseRowBatch:
    """Numeric labels inspected by task execution and diagnostics.

    Columns are packed index, request ordinal, token offset, phase (decode=0/prefill=1),
    identity-table index, prompt length and retained-replay flag. Tap maps index the
    allocated slot vector; index zero supplies the same unused-row fill as before.
    """

    step_id: int
    columns: Slots
    request_identities: tuple[RequestIdentity, ...]
    request_indices: Slots
    tap_selected: tuple[Slots, ...]
    tap_ranks: tuple[Slots, ...]
    identity_data: Slots
    identity_offsets: Slots

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PackedPhaseRowBatch) and self.step_id == other.step_id and self.rows == other.rows

    def __hash__(self) -> int:
        return hash((self.step_id, self.total_rows, tuple(self.rows)))

    @classmethod
    def from_rows(cls, step_id: int, total_rows: int, rows: tuple[PackedPhaseRow, ...]) -> PackedPhaseRowBatch:
        """Import explicit records for diagnostics and shape tracing."""
        assert total_rows == len(rows)
        identities = tuple(dict.fromkeys(row.identity for row in rows))
        lookup = {identity: index for index, identity in enumerate(identities)}
        columns = (
            np.array(
                [
                    (
                        r.packed_row_index,
                        r.request_ordinal,
                        r.token_offset,
                        {HiddenPhase.DECODE: 0, HiddenPhase.PREFILL: 1, HiddenPhase.NONE: -1}[r.phase],
                        lookup[r.identity],
                        r.prompt_length,
                        int(r.recomputed),
                    )
                    for r in rows
                ],
                dtype=np.int64,
            )
            .reshape(-1, 7)
            .T.copy()
        )
        last = {r.identity: index for index, r in enumerate(rows)}
        requests = np.fromiter(last.values(), dtype=np.int64)
        ordinal_last = np.fromiter({r.request_ordinal: index for index, r in enumerate(rows)}.values(), dtype=np.int64)
        fresh = columns[6] == 0
        prefill = np.flatnonzero((columns[3] == 1) & (columns[2] == columns[5] - 1) & fresh)
        decode = np.sort(ordinal_last[(columns[3, ordinal_last] == 0) & fresh[ordinal_last]])
        selected = (decode, prefill, np.concatenate((decode, prefill)))
        ranks = tuple(np.zeros(total_rows, dtype=np.int64) for _ in selected)
        for indices, mapping in zip(selected, ranks, strict=True):
            mapping[indices] = np.arange(len(indices))
        words = tuple(identity_words(identity) for identity in identities)
        offsets = np.cumsum([0, *(len(word) for word in words)], dtype=np.int64)
        return cls(
            step_id,
            columns,
            identities,
            requests,
            selected,
            ranks,
            np.concatenate(words) if words else np.empty(0, dtype=np.int64),
            offsets,
        )

    @property
    def total_rows(self) -> int:
        return self.columns.shape[1]

    def inspect(self, indices: Slots) -> tuple[PackedPhaseRow, ...]:
        """Materialize exact row records for an explicit consumer."""
        return tuple(
            PackedPhaseRow(
                packed,
                request,
                offset,
                HiddenPhase.PREFILL if phase == 1 else HiddenPhase.DECODE if phase == 0 else HiddenPhase.NONE,
                self.request_identities[int(identity)],
                prompt,
                bool(recomputed),
            )
            for packed, request, offset, phase, identity, prompt, recomputed in self.columns[:, indices].T.tolist()
        )

    def inspect_row(self, index: int) -> PackedPhaseRow:
        """Inspect one packed row without materializing the rest of the batch."""
        return self.inspect(np.array([index], dtype=np.int64))[0]

    def identity(self, index: int) -> RequestIdentity:
        """Borrow one row's admitted identity without constructing a row record."""
        return self.request_identities[int(self.columns[4, int(index)])]

    @property
    def rows(self) -> PackedRows:
        return PackedRows(self, np.arange(self.total_rows, dtype=np.int64))

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        return tuple(self.request_identities[int(index)] for index in self.columns[4])

    @property
    def request_rows(self) -> PackedRows:
        """Each request's final row in packed request order."""
        return PackedRows(self, self.request_indices)

    def producer_rows(self, phase: HiddenPhase | None) -> PackedRows:
        """Fresh prompt endings or latest decode rows."""
        if phase is None:
            return PackedRows(self, self.request_indices[self.columns[6, self.request_indices] == 0])
        return PackedRows(self, self.tap_selected[int(phase is HiddenPhase.PREFILL)])

    @property
    def prefill_rows(self) -> PackedRows:
        return PackedRows(self, np.flatnonzero(self.columns[3] == 1))

    @property
    def decode_rows(self) -> PackedRows:
        return PackedRows(self, np.flatnonzero(self.columns[3] == 0))

    @property
    def latest_decode_rows(self) -> PackedRows:
        """Each request's final fresh decode row."""
        return PackedRows(self, self.tap_selected[0])


@dataclass(frozen=True, slots=True)
class PackedSamplerRowLink:
    """One exact sampler-row to packed-model-row relation."""

    sampler_row_index: int
    packed_row_index: int


@dataclass(frozen=True, slots=True, eq=False)
class PackedSamplerRowBatch:
    """Ordered sampler indices into one exact numeric packed row batch."""

    row_batch: PackedPhaseRowBatch
    packed_indices: Slots

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PackedSamplerRowBatch) and self.row_batch == other.row_batch and self.links == other.links
        )

    def __hash__(self) -> int:
        return hash((self.row_batch, self.links))

    @classmethod
    def from_links(
        cls, row_batch: PackedPhaseRowBatch, links: tuple[PackedSamplerRowLink, ...]
    ) -> PackedSamplerRowBatch:
        """Import explicit sampler links for diagnostics and shape tracing."""
        indices = np.empty(len(links), dtype=np.int64)
        for link in links:
            indices[link.sampler_row_index] = link.packed_row_index
        return cls(row_batch, indices)

    @property
    def links(self) -> tuple[PackedSamplerRowLink, ...]:
        """Materialize sampler links for explicit inspection."""
        return tuple(PackedSamplerRowLink(index, packed) for index, packed in enumerate(self.packed_indices.tolist()))

    @property
    def step_id(self) -> int:
        return self.row_batch.step_id

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        return tuple(
            self.row_batch.request_identities[int(index)] for index in self.row_batch.columns[4, self.packed_indices]
        )


class RowLabeler:
    """Admission-time identities and native frontier ownership for one session."""

    def __init__(self) -> None:
        self.native = _core.RowLabeler()
        self.ids: dict[str, int] = {}
        self.identities = np.empty(0, dtype=np.object_)
        self.slots = np.empty(0, dtype=np.int64)

    def admit(self, identity: RequestIdentity) -> None:
        """Intern a newly admitted request generation."""
        index = self.native.admit(identity_words(identity))
        self.ids[identity.request_id] = index
        if index >= len(self.identities):
            identities = np.empty(max(16, 2 * index), dtype=np.object_)
            identities[: len(self.identities)] = self.identities
            self.identities = identities
        self.identities[index] = identity

    def retire(self, identity: RequestIdentity) -> None:
        """Release an admitted request generation and its retained frontier."""
        index = self.ids.pop(identity.request_id)
        self.native.retire(index)
        self.identities[index] = None

    def bind(self, request_id: str, slot: int, capacity: int) -> None:
        """Bind MRv2's persistent slot when vLLM admits or resumes a request."""
        if len(self.slots) != capacity:
            self.slots = np.full(capacity, -1, dtype=np.int64)
        self.slots[slot] = self.ids[request_id]

    def prepare(
        self, scheduled: np.ndarray, prompt: np.ndarray, computed: np.ndarray, ids: Slots, step_id: int, keep: bool
    ) -> PackedSamplerRowBatch:
        """Borrow contiguous host columns for one native label and hidden-map operation."""
        columns, links, selected, ranks, words, offsets = self.native.prepare(scheduled, prompt, computed, ids, keep)
        rows = PackedPhaseRowBatch(
            step_id,
            columns,
            tuple(self.identities[ids]),
            links[scheduled != 0],
            tuple(selected),
            tuple(ranks),
            words,
            offsets,
        )
        return PackedSamplerRowBatch(rows, links)


def label_rows(
    identities: tuple[RequestIdentity, ...],
    scheduled: list[int],
    prompt: list[int],
    computed: list[int],
    step_id: int,
) -> PackedSamplerRowBatch:
    """Label explicit host input for CPU checks using the runner's bulk implementation."""
    labeler = RowLabeler()
    for identity in identities:
        labeler.admit(identity)
    return labeler.prepare(
        np.asarray(scheduled, dtype=np.int32),
        np.asarray(prompt, dtype=np.int32),
        np.asarray(computed, dtype=np.int32),
        np.arange(len(identities), dtype=np.int64),
        step_id,
        False,
    )


__all__ = [
    "label_rows",
    "RowLabeler",
    "PackedRows",
    "PackedPhaseRow",
    "PackedPhaseRowBatch",
    "PackedSamplerRowBatch",
    "PackedSamplerRowLink",
]
