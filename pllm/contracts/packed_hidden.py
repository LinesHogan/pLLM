"""Hidden resource bindings and slot-addressed publications passed to tensor consumers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
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
class HiddenRows:
    """One module's immutable publication, addressed within its stable slot storage."""

    resource: PackedHiddenResourceBinding
    storage: torch.Tensor
    # Dense packed-row -> slot lookup. Uncaptured rows address a valid padding slot.
    slots: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class HiddenRowBatch:
    """Resource-keyed slot publications for one exact packed row batch."""

    row_batch: PackedPhaseRowBatch
    captures: tuple[HiddenRows, ...]

    @property
    def step_id(self) -> int:
        return self.row_batch.step_id


__all__ = [
    "PackedHiddenResourceBinding",
    "HiddenRows",
    "HiddenRowBatch",
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
