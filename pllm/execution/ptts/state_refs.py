"""Array-backed version selections for PTTS; individual identities are materialized on inspection."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import SupportsIndex, overload

import numpy as np
import numpy.typing as npt

from pllm import _core
from pllm.contracts.state_ref import LayoutId, LineageId, StateRef, VersionId

Indices = npt.NDArray[np.int64]


@dataclass(frozen=True, eq=False)
class StateRefs(Sequence[StateRef]):
    """Ordered identities, optionally resolved to arena rows; arrays never change after issue.

    Layouts normally repeat one ID. Keeping a column also preserves the order of atomic
    admissions and selections spanning several resources without constructing individual refs.
    Rows are hints only: the arena checks lineage identity before using a recycled table row.
    """

    layouts: Indices
    lineages: Indices
    versions: Indices
    rows: Indices | None = None

    def __post_init__(self) -> None:
        for column in (self.layouts, self.lineages, self.versions, self.rows):
            if column is not None:
                column.flags.writeable = False

    def __len__(self) -> int:
        return len(self.lineages)

    def __iter__(self) -> Iterator[StateRef]:
        return iter(self.refs)

    @overload
    def __getitem__(self, index: SupportsIndex) -> StateRef: ...

    @overload
    def __getitem__(self, index: slice) -> StateRefs: ...

    def __getitem__(self, index: SupportsIndex | slice) -> StateRef | StateRefs:
        if isinstance(index, slice):
            return self.take(index)
        index = int(index)
        return StateRef(
            LayoutId(int(self.layouts[index])),
            LineageId(int(self.lineages[index])),
            VersionId(int(self.versions[index])),
        )

    @property
    def refs(self) -> tuple[StateRef, ...]:
        """The individual identities, built only when a cold path inspects them."""
        return tuple(
            StateRef(LayoutId(layout), LineageId(lineage), VersionId(version))
            for layout, lineage, version in zip(self.layouts.tolist(), self.lineages.tolist(), self.versions.tolist())
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateRefs):
            return NotImplemented
        return self is other or _core.refs_equal(
            self.layouts, self.lineages, self.versions, other.layouts, other.lineages, other.versions
        )

    @classmethod
    def of(cls, refs: Sequence[StateRef]) -> StateRefs:
        """Convert cold individual refs into an array selection at the caller boundary."""
        return cls(
            np.fromiter((ref.layout_id for ref in refs), np.int64, len(refs)),
            np.fromiter((ref.lineage_id for ref in refs), np.int64, len(refs)),
            np.fromiter((ref.version_id for ref in refs), np.int64, len(refs)),
        )

    def resolved(self, rows: Indices) -> StateRefs:
        """The same identities with their arena rows attached."""
        return StateRefs(self.layouts, self.lineages, self.versions, rows)

    @classmethod
    def join(cls, groups: Sequence[StateRefs]) -> StateRefs:
        """Concatenate resource selections, preserving their order."""
        if len(groups) == 1:
            return groups[0]
        if not groups:
            return cls.of(())
        rows = (
            np.concatenate([group.rows for group in groups if group.rows is not None])
            if all(group.rows is not None for group in groups)
            else None
        )
        return cls(
            np.concatenate([group.layouts for group in groups]),
            np.concatenate([group.lineages for group in groups]),
            np.concatenate([group.versions for group in groups]),
            rows,
        )

    def take(self, indices: Sequence[int] | Indices | slice) -> StateRefs:
        """Select a batch without materializing identities."""
        index = indices if isinstance(indices, (np.ndarray, slice)) else np.asarray(indices, dtype=np.int64)
        return StateRefs(
            np.ascontiguousarray(self.layouts[index]),
            np.ascontiguousarray(self.lineages[index]),
            np.ascontiguousarray(self.versions[index]),
            None if self.rows is None else np.ascontiguousarray(self.rows[index]),
        )

    def positions(self, refs: StateRefs) -> Indices:
        """Positions of exact identities in this batch; -1 denotes an absent version."""
        return _core.ref_positions(
            self.layouts, self.lineages, self.versions, refs.layouts, refs.lineages, refs.versions
        )

    def replace_versions(self, before: StateRefs, after: StateRefs) -> StateRefs:
        """Substitute a transition's exact sources while retaining selection order."""
        positions = before.positions(self)
        return StateRefs(
            self.layouts, self.lineages, _core.replace_versions(self.versions, positions, after.versions), self.rows
        )

    def next_versions(self) -> StateRefs:
        """The successor of every selected version, in the same arena rows."""
        return StateRefs(self.layouts, self.lineages, self.versions + 1, self.rows)
