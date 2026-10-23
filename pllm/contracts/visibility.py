"""State read policies for authors and checked read-from relations for the planner."""

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
    """Internal equation identity and read domain, before physical partitioning.

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
    kv: Literal["causal_history"] = "causal_history"
    sweeps: int = 1


@dataclass(frozen=True, slots=True)
class StateVisibility:
    """Rust-derived reader/writer maps; no physical slots or mutable execution state."""

    source: VisibilitySource
    reads: str
    dependencies: str

    def explain(self) -> str:
        """List symbolic model/update readers and their logical writers."""
        status = " (planned only)" if self.source.scope == "overlay" and self.source.policy == Exact() else ""
        return (
            f"{self.source.scope}: {self.source.policy.kind}{status}, sweeps={self.source.sweeps}\n"
            "r=request, t=target, c=chunk; entry is pinned, update[c] writes boundary c+1\n"
            + self.reads.replace("; ", ";\n")
        )
