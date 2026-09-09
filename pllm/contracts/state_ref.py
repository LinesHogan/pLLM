"""Logical identities for versioned test-time tensor state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType


LayoutId = NewType("LayoutId", int)
LineageId = NewType("LineageId", int)
VersionId = NewType("VersionId", int)


@dataclass(frozen=True, slots=True, order=True)
class StateRef:
    """Representation-independent logical state version identity."""

    layout_id: LayoutId
    lineage_id: LineageId
    version_id: VersionId

    def __post_init__(self) -> None:
        if int(self.layout_id) < 0:
            raise ValueError("state layout id must be non-negative")
        if int(self.lineage_id) < 1:
            raise ValueError("state lineage id must be positive")
        if int(self.version_id) < 0:
            raise ValueError("state version id must be non-negative")


__all__ = ["LayoutId", "LineageId", "StateRef", "VersionId"]
