"""Authoring names and timing choices retained for workload diagnostics and composition."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemberDeclaration:
    """A method or model value, before planning chooses its task placement."""

    effect: str
    phase: str | None = None
    tokens: str | None = None
    visibility: str | None = None
