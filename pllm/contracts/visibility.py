"""State read policies for authors and the read domain the planner derives relations from."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Visibility:
    """Choose which state a model read sees; use Exact or forward_pinned."""

    kind: Literal["forward_pinned", "exact"]

    @staticmethod
    def forward_pinned() -> "Visibility":
        """Keep all model reads on the state selected at forward entry."""
        return Visibility("forward_pinned")


def Exact() -> Visibility:
    """Read the state produced by every preceding logical chunk."""
    return Visibility("exact")


@dataclass(frozen=True, slots=True)
class VisibilitySource:
    """Internal equation identity and read domain; Rust derives the read-from map.

    A pinned overlay exposes one whole-forward transition. Its private chunk/tail
    mathematics remain in the equation digest; chunk_size=0 means no model chunk
    boundary is exposed. Exact sources instead expose the logical chunk domain.
    """

    policy: Visibility
    scope: Literal["overlay", "model_scan"]
    equations_digest: str
    resources: tuple[str, ...]
    chunk_size: int
    tail: Literal["transition", "include", "carry"]
