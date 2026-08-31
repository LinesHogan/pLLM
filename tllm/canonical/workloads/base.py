"""Nominal marker for clean-room workload routing."""

from __future__ import annotations


class CanonicalWorkload:
    """Nominal base used only by the outer runtime routing boundary."""

    __slots__ = ()


__all__ = ["CanonicalWorkload"]
