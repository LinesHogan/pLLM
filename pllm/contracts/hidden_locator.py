"""Closed hidden-activation locator vocabulary shared across compiler layers."""

from __future__ import annotations

from enum import StrEnum


class HiddenLayerKind(StrEnum):
    """Closed hidden-layer locator forms shared by author IR and schedules."""

    NONE = "none"
    INDEX = "index"
    PATH = "path"
    WILDCARD = "wildcard"


class HiddenSite(StrEnum):
    """Signed hidden-activation tap sites.

    ``NONE`` is the only non-hidden wire value and serializes as the legacy
    empty string.
    """

    NONE = ""
    POST = "post"
    BLOCK_OUTPUT = "block_output"
    MODEL_OUTPUT = "model_output"


class HiddenPhase(StrEnum):
    """Signed hidden-activation execution phases."""

    NONE = ""
    PREFILL = "prefill"
    DECODE = "decode"


__all__ = [
    "HiddenLayerKind",
    "HiddenPhase",
    "HiddenSite",
]
