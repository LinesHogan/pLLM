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


class HiddenPhase(StrEnum):
    """Signed hidden-activation execution phases."""

    NONE = ""
    PREFILL = "prefill"
    DECODE = "decode"


class HiddenLocatorCapability(StrEnum):
    """Backend-declared hidden locator bindings admitted by a timeline."""

    POST_PREFILL = "hidden/post/prefill"
    POST_DECODE = "hidden/post/decode"
    BLOCK_OUTPUT_PREFILL = "hidden/block_output/prefill"
    BLOCK_OUTPUT_DECODE = "hidden/block_output/decode"

    @classmethod
    def for_locator(
        cls,
        site: HiddenSite | str,
        phase: HiddenPhase | str,
    ) -> "HiddenLocatorCapability":
        normalized_site = HiddenSite(site)
        normalized_phase = HiddenPhase(phase)
        if normalized_site is HiddenSite.NONE or normalized_phase is HiddenPhase.NONE:
            raise ValueError("hidden locator capability requires a site and phase")
        return cls(f"hidden/{normalized_site.value}/{normalized_phase.value}")


__all__ = [
    "HiddenLayerKind",
    "HiddenLocatorCapability",
    "HiddenPhase",
    "HiddenSite",
]
