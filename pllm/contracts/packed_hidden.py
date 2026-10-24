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

    def __post_init__(self) -> None:
        if not self.snapshots:
            raise ValueError("packed hidden snapshots must be non-empty")
        locator_keys = tuple(snapshot.resource.locator_key for snapshot in self.snapshots)
        if tuple(sorted(locator_keys)) != locator_keys or len(set(locator_keys)) != len(locator_keys):
            raise ValueError("packed hidden snapshots must have sorted unique resources")
        reference = self.snapshots[0].hidden
        if reference.ndim != 2 or int(reference.shape[0]) != self.row_batch.total_rows:
            raise ValueError("packed hidden snapshots must align with one row batch")
        if any(
            snapshot.hidden.shape != reference.shape
            or snapshot.hidden.dtype is not reference.dtype
            or snapshot.hidden.device != reference.device
            or torch.is_inference(snapshot.hidden)
            for snapshot in self.snapshots
        ):
            raise ValueError("packed hidden snapshots must be aligned ordinary tensors")

    @property
    def plan_epoch(self) -> int:
        return self.row_batch.plan_epoch

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
    if site is not HiddenSite.BLOCK_OUTPUT:
        raise ValueError("Qwen2 hidden input requires a block or final model output")
    module_index = layer if layer >= 0 else layer_count + layer
    if module_index < 0 or module_index >= layer_count:
        raise ValueError("hidden layer does not identify a Qwen2 block")
    return PackedHiddenResourceBinding(
        str(layer), HiddenLayerKind.INDEX, HiddenSite.BLOCK_OUTPUT, f"model.layers.{module_index}"
    )
