"""State read policies for authors and the read domain the planner derives relations from."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Visibility:
    """Choose which state a model read sees; use Exact or forward_pinned."""

    kind: Literal["forward_pinned", "exact", "step_lag"]

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

    Overlays keep their chunk/tail mathematics inside the transition's own scan
    (chunk_size=0); ModelScan exposes its logical chunk domain to the planner.
    """

    policy: Visibility
    scope: Literal["overlay", "model_scan", "post_sampler"]
    equations_digest: str
    resources: tuple[str, ...]
    chunk_size: int
    tail: Literal["transition", "include", "carry"]
    lag: int = 0

    @staticmethod
    def post_sampler(lag: int) -> "VisibilitySource":
        """Declare the retained-input boundary for the optional serial writer lane."""
        return VisibilitySource(
            Visibility("step_lag"), "post_sampler", "retained-step-inputs@1", (), 0, "transition", lag
        )

    @property
    def in_forward(self) -> bool:
        """An Exact overlay runs its transition at the projection, inside the model forward."""
        return self.scope == "overlay" and self.policy == Exact()
