"""State read policies for authors and the read domain the planner derives relations from."""

from dataclasses import dataclass
from typing import Literal

# Which state version a model read sees: "forward_pinned" the version the forward started with (later updates
# only behind explicit route boundaries), "exact" also every chunk trained before it, "step_lag" the version
# published a fixed number of steps earlier (the post-sampler writer lane).
VisibilityKind = Literal["forward_pinned", "exact", "step_lag"]


@dataclass(frozen=True, slots=True)
class VisibilitySource:
    """A read policy and its read domain; Rust derives the read-from map.

    Overlays keep their chunk/tail mathematics inside the transition's own scan (chunk_size=0).
    """

    policy: VisibilityKind
    scope: Literal["overlay", "post_sampler"]
    resources: tuple[str, ...]
    chunk_size: int
    tail: Literal["transition", "carry"]
    lag: int = 0

    @staticmethod
    def post_sampler(lag: int) -> "VisibilitySource":
        """Declare the retained-input boundary for the optional serial writer lane."""
        return VisibilitySource("step_lag", "post_sampler", (), 0, "transition", lag)

    @property
    def in_forward(self) -> bool:
        """Exact entry reads and ordinary projection updates use accumulated private state."""
        return self.scope == "overlay" and self.policy == "exact"
