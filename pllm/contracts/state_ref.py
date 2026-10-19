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


__all__ = ["LayoutId", "LineageId", "StateRef", "VersionId"]
