"""Typed packed-row facts shared across backend adapters and runtime services."""

from __future__ import annotations

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import overload

import numpy as np
import numpy.typing as npt

from pllm import _core
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.request_identity import RequestIdentity

Slots = npt.NDArray[np.int64]


class TapRows(IntEnum):
    """Which of PackedPhaseRowBatch.tap_selected a tap keeps, in the native labeler's order."""

    LATEST_DECODE = 0
    PROMPT_ENDINGS = 1
    DECODE_AND_PROMPT_ENDINGS = 2
    FRESH = 3  # every token computed fresh this step, a superset of the others


def tap_selection(phases: Collection[HiddenPhase]) -> TapRows:
    """The rows a tap reading `phases` keeps."""
    if HiddenPhase.ALL in phases:
        return TapRows.FRESH
    if len(phases) == 2:
        return TapRows.DECODE_AND_PROMPT_ENDINGS
    return TapRows.PROMPT_ENDINGS if HiddenPhase.PREFILL in phases else TapRows.LATEST_DECODE


@dataclass(frozen=True, slots=True)
class PackedPhaseRow:
    """One packed row with its stable request identity, read from a PackedPhaseRowBatch; batches come only from
    the native RowLabeler (label_rows for explicit scheduler counts)."""

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

    def __iter__(self) -> Iterator[PackedPhaseRow]:
        return iter(self.batch.inspect(self.indices))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PackedRows) and self.batch is other.batch:
            return np.array_equal(self.indices, other.indices)
        return tuple(self) == other

    @property
    def identity_indices(self) -> Slots:
        """Each row's index into batch.request_identities."""
        return self.batch.identity_indices[self.indices]

    @property
    def token_offsets(self) -> Slots:
        return self.batch.token_offsets[self.indices]

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        """Borrow admitted identities in row order without constructing row records."""
        identities = np.asarray(self.batch.request_identities, dtype=np.object_)
        return tuple(identities[self.identity_indices])

    def producer(self, phase: HiddenPhase | None) -> PackedRows:
        """Select a producer phase without inspecting records."""
        if phase is None:
            return self
        selected = self.batch.tap_selected[tap_selection((phase,))]
        return PackedRows(self.batch, self.indices[np.isin(self.indices, selected)])


@dataclass(frozen=True, slots=True, eq=False)
class PackedPhaseRowBatch:
    """Numeric labels inspected by task execution and diagnostics.

    The rows of `columns` are named by the properties below (the native labeler writes the same order);
    `columns[0]` is the packed index. Tap selections are ordered as in TapRows; tap maps index the
    allocated slot vector, and index zero fills unselected rows.
    """

    step_id: int
    columns: Slots
    request_identities: tuple[RequestIdentity, ...]
    request_indices: Slots
    tap_selected: tuple[Slots, ...]
    tap_ranks: tuple[Slots, ...]
    identity_data: Slots
    identity_offsets: Slots

    @property
    def total_rows(self) -> int:
        return self.columns.shape[1]

    @property
    def request_ordinals(self) -> Slots:
        """Each row's request position in this step's scheduling order."""
        return self.columns[1]

    @property
    def token_offsets(self) -> Slots:
        return self.columns[2]

    @property
    def phases(self) -> Slots:
        """1 for prompt (prefill) rows, 0 for decode rows, -1 for rows without a phase."""
        return self.columns[3]

    @property
    def identity_indices(self) -> Slots:
        """Each row's index into request_identities."""
        return self.columns[4]

    @property
    def prompt_lengths(self) -> Slots:
        return self.columns[5]

    @property
    def recomputed(self) -> Slots:
        """1 where retained state already consumed the token (retained replay), else 0."""
        return self.columns[6]

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
        return self.request_identities[int(self.identity_indices[int(index)])]

    @property
    def rows(self) -> PackedRows:
        return PackedRows(self, np.arange(self.total_rows, dtype=np.int64))

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        return tuple(self.request_identities[int(index)] for index in self.identity_indices)

    @property
    def request_rows(self) -> PackedRows:
        """Each request's final row in packed request order."""
        return PackedRows(self, self.request_indices)

    def producer_rows(self, phase: HiddenPhase | None) -> PackedRows:
        """Fresh prompt endings, latest decode rows or every fresh token; None: each request's fresh final row."""
        if phase is None:
            return PackedRows(self, self.request_indices[self.recomputed[self.request_indices] == 0])
        return PackedRows(self, self.tap_selected[tap_selection((phase,))])

    @property
    def latest_decode_rows(self) -> PackedRows:
        """Each request's final fresh decode row."""
        return PackedRows(self, self.tap_selected[TapRows.LATEST_DECODE])


@dataclass(frozen=True, slots=True, eq=False)
class PackedSamplerRowBatch:
    """Ordered sampler indices into one exact numeric packed row batch."""

    row_batch: PackedPhaseRowBatch
    packed_indices: Slots

    @property
    def step_id(self) -> int:
        return self.row_batch.step_id

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        return tuple(
            self.row_batch.request_identities[int(index)]
            for index in self.row_batch.identity_indices[self.packed_indices]
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
    "tap_selection",
    "TapRows",
    "RowLabeler",
    "PackedRows",
    "PackedPhaseRow",
    "PackedPhaseRowBatch",
    "PackedSamplerRowBatch",
]
