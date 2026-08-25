"""Runtime-local versioned request state with owned Tensor buffers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import torch


@dataclass(slots=True)
class VersionedStateSlot:
    slot_id: int
    key: Hashable
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    version: int = 0
    step: int = 0


def _owned_tensor(value: torch.Tensor) -> torch.Tensor:
    if torch.is_inference_mode_enabled() or value.is_inference():
        with torch.inference_mode(False):
            return value.detach().clone()
    return value.detach().clone()


class VersionedRequestStateBank:
    """Small neutral bank for runtime-owned request-keyed state."""

    def __init__(
        self,
        name: str,
        *,
        max_slots: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        self.name = str(name)
        self._slots: dict[Hashable, VersionedStateSlot] = {}
        self._next_slot_id = 1
        self._max_slots = max_slots
        self._max_bytes = max_bytes

    def set_limits(self, *, max_slots: int | None, max_bytes: int | None) -> None:
        self._max_slots = max_slots
        self._max_bytes = max_bytes

    def slot_for_key(self, key: Hashable) -> VersionedStateSlot:
        slot = self._slots.get(key)
        if slot is not None:
            return slot
        if self._max_slots is not None and len(self._slots) >= self._max_slots:
            raise RuntimeError(
                f"state bank {self.name!r} slot limit exceeded: "
                f"slots={len(self._slots) + 1} max_slots={self._max_slots}"
            )
        slot = VersionedStateSlot(slot_id=self._next_slot_id, key=key)
        self._next_slot_id += 1
        self._slots[key] = slot
        return slot

    def get_slot(self, key: Hashable) -> VersionedStateSlot | None:
        return self._slots.get(key)

    def keys(self) -> tuple[Hashable, ...]:
        return tuple(self._slots)

    def clear(self, key: Hashable) -> None:
        self._slots.pop(key, None)

    @staticmethod
    def _tensor_bytes(value: torch.Tensor) -> int:
        return int(value.numel() * value.element_size())

    @property
    def estimated_bytes(self) -> int:
        return sum(
            self._tensor_bytes(value)
            for slot in self._slots.values()
            for value in slot.buffers.values()
        )

    def publish_buffers(
        self,
        slot: VersionedStateSlot,
        buffers: dict[str, torch.Tensor],
    ) -> int:
        owned = {name: _owned_tensor(value) for name, value in buffers.items()}
        old_bytes = sum(self._tensor_bytes(value) for value in slot.buffers.values())
        new_bytes = sum(self._tensor_bytes(value) for value in owned.values())
        next_bytes = self.estimated_bytes - old_bytes + new_bytes
        if self._max_bytes is not None and next_bytes > self._max_bytes:
            raise RuntimeError(
                f"state bank {self.name!r} byte limit exceeded: "
                f"bytes={next_bytes} max_bytes={self._max_bytes}"
            )
        slot.buffers = owned
        slot.version += 1
        slot.step += 1
        return slot.version
