"""Typed provenance and payloads for eager packed hidden snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tllm.contracts.hidden_locator import HiddenLayerKind, HiddenSite
from tllm.contracts.packed_rows import PackedPhaseRowBatch


def _packed_hidden_wire_name(value: object, label: str) -> str:
    """Parse one exact string at the Python/Rust manifest boundary."""

    if type(value) is str:
        if value:
            return value
        raise ValueError(f"{label} must be non-empty")
    raise TypeError(f"{label} must be str")


@dataclass(frozen=True, slots=True)
class PackedHiddenResourceBinding:
    """One signed backend module for a packed block-output locator."""

    layer: str
    layer_kind: HiddenLayerKind
    site: HiddenSite
    module_path: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "layer",
            _packed_hidden_wire_name(
                self.layer,
                "PackedHiddenResourceBinding.layer",
            ),
        )
        object.__setattr__(
            self,
            "layer_kind",
            HiddenLayerKind(self.layer_kind),
        )
        object.__setattr__(self, "site", HiddenSite(self.site))
        if self.layer_kind not in (
            HiddenLayerKind.INDEX,
            HiddenLayerKind.PATH,
        ):
            raise ValueError(
                "PackedHiddenResourceBinding.layer_kind must be index or path"
            )
        if self.site is not HiddenSite.BLOCK_OUTPUT:
            raise ValueError(
                "PackedHiddenResourceBinding.site must be block_output"
            )
        if self.layer_kind is HiddenLayerKind.INDEX:
            try:
                layer_index = int(self.layer)
            except ValueError as error:
                raise ValueError(
                    "PackedHiddenResourceBinding INDEX layer must be a canonical "
                    "signed integer"
                ) from error
            if (
                str(layer_index) != self.layer
                or layer_index < -(1 << 63)
                or layer_index >= (1 << 63)
            ):
                raise ValueError(
                    "PackedHiddenResourceBinding INDEX layer must be a canonical "
                    "signed 64-bit integer"
                )
        object.__setattr__(
            self,
            "module_path",
            _packed_hidden_wire_name(
                self.module_path,
                "PackedHiddenResourceBinding.module_path",
            ),
        )

    @property
    def locator_key(self) -> tuple[str, str, str]:
        return (self.layer, self.layer_kind.value, self.site.value)


@dataclass(frozen=True, slots=True)
class PackedHiddenSnapshotSpecialization:
    """Cold facts for one fixed-capacity eager snapshot service."""

    plan_epoch: int
    max_packed_rows: int
    hidden_size: int
    resources: tuple[PackedHiddenResourceBinding, ...]
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        if self.plan_epoch < 1:
            raise ValueError("packed hidden plan epoch must be positive")
        if self.max_packed_rows < 1:
            raise ValueError("max packed rows must be positive")
        if self.hidden_size < 1:
            raise ValueError("packed hidden size must be positive")
        if not self.resources:
            raise ValueError("packed hidden resources must be non-empty")
        locator_keys = tuple(
            resource.locator_key for resource in self.resources
        )
        if tuple(sorted(locator_keys)) != locator_keys or len(
            set(locator_keys)
        ) != len(locator_keys):
            raise ValueError(
                "packed hidden resources must have sorted unique locators"
            )


@dataclass(frozen=True, slots=True)
class PackedHiddenSnapshot:
    """One exact signed resource and its owned ordinary hidden tensor."""

    resource: PackedHiddenResourceBinding
    hidden: torch.Tensor


@dataclass(frozen=True, slots=True)
class PackedHiddenSnapshotBatch:
    """Owned resource-keyed tensors for one exact packed row batch."""

    row_batch: PackedPhaseRowBatch
    snapshots: tuple[PackedHiddenSnapshot, ...]

    def __post_init__(self) -> None:
        if not self.snapshots:
            raise ValueError("packed hidden snapshots must be non-empty")
        locator_keys = tuple(
            snapshot.resource.locator_key for snapshot in self.snapshots
        )
        if tuple(sorted(locator_keys)) != locator_keys or len(
            set(locator_keys)
        ) != len(locator_keys):
            raise ValueError(
                "packed hidden snapshots must have sorted unique resources"
            )
        reference = self.snapshots[0].hidden
        if reference.ndim != 2 or int(reference.shape[0]) != self.row_batch.total_rows:
            raise ValueError(
                "packed hidden snapshots must align with one row batch"
            )
        if any(
            snapshot.hidden.shape != reference.shape
            or snapshot.hidden.dtype is not reference.dtype
            or snapshot.hidden.device != reference.device
            or torch.is_inference(snapshot.hidden)
            for snapshot in self.snapshots
        ):
            raise ValueError(
                "packed hidden snapshots must be aligned ordinary tensors"
            )

    @property
    def plan_epoch(self) -> int:
        return self.row_batch.plan_epoch

    @property
    def step_id(self) -> int:
        return self.row_batch.step_id

    def hidden_for(
        self, resource: PackedHiddenResourceBinding
    ) -> torch.Tensor:
        for snapshot in self.snapshots:
            if snapshot.resource == resource:
                return snapshot.hidden
        raise KeyError(resource)


__all__ = [
    "PackedHiddenResourceBinding",
    "PackedHiddenSnapshot",
    "PackedHiddenSnapshotBatch",
    "PackedHiddenSnapshotSpecialization",
]
