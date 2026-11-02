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
    """One signed backend module for a packed hidden locator, and which of its outputs the tap copies: 0 for a
    layer's fresh output or a norm's normalized value, 1 for the residual stream a norm returns beside it."""

    layer: str
    layer_kind: HiddenLayerKind
    site: HiddenSite
    module_path: str
    output: int = 0

    @property
    def locator_key(self) -> tuple[str, str, str]:
        return (self.layer, self.layer_kind.value, self.site.value)

    @property
    def tap(self) -> tuple[str, int]:
        """The module output a tap copies rows from."""
        return (self.module_path, self.output)


@dataclass(frozen=True, slots=True)
class HiddenRows:
    """One module's immutable publication, addressed within its stable slot storage."""

    resource: PackedHiddenResourceBinding
    storage: torch.Tensor
    # Dense packed-row -> slot lookup. Rows without a tap address a valid padding slot.
    slots: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class HiddenRowBatch:
    """Resource-keyed slot publications for one exact packed row batch."""

    row_batch: PackedPhaseRowBatch
    taps: tuple[HiddenRows, ...]

    @property
    def step_id(self) -> int:
        return self.row_batch.step_id


__all__ = [
    "PackedHiddenResourceBinding",
    "HiddenRows",
    "HiddenRowBatch",
]
