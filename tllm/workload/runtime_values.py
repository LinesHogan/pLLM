"""Closed value markers shared without importing the runtime package."""

from enum import Enum


class RuntimeValueMarker(Enum):
    SKIPPED = "skipped"


SKIPPED = RuntimeValueMarker.SKIPPED

__all__ = ["RuntimeValueMarker", "SKIPPED"]
