"""Typed provenance and payloads for eager packed hidden snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from pllm.contracts.packed_rows import PackedPhaseRowBatch


@dataclass(frozen=True, slots=True)
class PackedHiddenResourceBinding:
    """One signed backend module for a packed block-output locator."""

    layer: str
    layer_kind: HiddenLayerKind
    site: HiddenSite
    module_path: str

    @property
    def locator_key(self) -> tuple[str, str, str]:
        return (self.layer, self.layer_kind.value, self.site.value)


@dataclass(frozen=True, slots=True)
class PackedHiddenSnapshot:
    """One hidden resource's rows for this step.

    `hidden` holds the step's rows. `buffer`, when set, is the fixed-address tensor `hidden` is a
    leading-row view of (a model tap); captured steps read from it so replays see new rows.
    """

    resource: PackedHiddenResourceBinding
    hidden: torch.Tensor
    buffer: torch.Tensor | None = None

    @property
    def storage(self) -> torch.Tensor:
        """The fixed tensor to index from: the tap buffer, or the rows themselves."""
        return self.hidden if self.buffer is None else self.buffer


@dataclass(frozen=True, slots=True)
class PackedHiddenSnapshotBatch:
    """Owned resource-keyed tensors for one exact packed row batch."""

    row_batch: PackedPhaseRowBatch
    snapshots: tuple[PackedHiddenSnapshot, ...]

    @property
    def step_id(self) -> int:
        return self.row_batch.step_id


__all__ = [
    "PackedHiddenResourceBinding",
    "PackedHiddenSnapshot",
    "PackedHiddenSnapshotBatch",
]


def hidden_binding(layer: int, site: HiddenSite, layer_count: int) -> PackedHiddenResourceBinding:
    """The binding a Qwen2-style decoder exposes: block outputs by layer index, the final output as `model.norm`."""
    if site is HiddenSite.MODEL_OUTPUT and layer == -1:
        return PackedHiddenResourceBinding(str(layer), HiddenLayerKind.INDEX, site, "model.norm")
    module_index = layer if layer >= 0 else layer_count + layer
    if site is not HiddenSite.BLOCK_OUTPUT or not 0 <= module_index < layer_count:
        raise ValueError(f"hidden layer {layer} at site {site.value} is outside 0..{layer_count - 1}")
    return PackedHiddenResourceBinding(
        str(layer), HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers.{module_index}"
    )
