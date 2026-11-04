"""Static per-layer KV storage declarations shared by authoring, checks and vLLM."""

from dataclasses import dataclass, replace
from typing import Literal

Keep = Literal["all", "window", "none"]


@dataclass(frozen=True)
class KVLayout:
    """One layer's storage; source counts layers below, sink retains the entire prompt."""

    layer: int
    keep: Keep = "all"
    window: int = 0
    source: int = 0
    sink: bool = False

    @property
    def description(self) -> str:
        """Describe storage separately from a workload's optional read selection."""
        if self.keep == "none":
            return f"shared layer {self.layer - self.source} (no KV allocation)"
        if self.keep == "window":
            return f"{'prompt sink + ' if self.sink else ''}window {self.window}"
        return "all"


def expand_layouts(layouts: tuple[KVLayout, ...], layers: int) -> tuple[KVLayout, ...]:
    """Expand the all-layers declaration once model geometry is available."""
    return tuple(
        replace(layout, layer=layer)
        for layout in layouts
        for layer in (range(layers) if layout.layer == -1 else (layout.layer,))
    )


def storage_layout(layouts: tuple[KVLayout, ...], layer: int) -> KVLayout:
    """Resolve an earlier-layer sharing chain to its sole storage owner."""
    by_layer = {layout.layer: layout for layout in layouts}
    layout = by_layer.get(layer, KVLayout(layer))
    while layout.keep == "none":
        layer -= layout.source
        layout = by_layer.get(layer, KVLayout(layer))
    return layout


@dataclass(frozen=True)
class KVCapacity:
    """Native allocation upper bound per request, including in-flight scheduled tokens."""

    layout: KVLayout
    block_size: int
    blocks: int
    block_bytes: int

    @property
    def bytes(self) -> int:
        """Physical KV bytes per request at the native admission bound."""
        return self.blocks * self.block_bytes
