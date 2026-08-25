#!/usr/bin/env python3
"""Program state banks: versioned parameter/optimizer slots.

``REPLICA`` banks hold one shared slot. ``REQUEST`` banks hold one slot per
prompt identity; rows in a batch are grouped by prompt (first-appearance
order) so a duplicated prompt is updated exactly once per batch. Every write
publishes ``version + 1``; readers observe the version they read, which is
what makes the STATE fence auditable in the eager lane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import torch

from tllm.program.optim import OptState
from tllm.workgraph.frontend import (
    PromptIdxKey,
    RequestKeyField,
    RequestStateKey,
    StateScope,
)
from tllm.workgraph.lowering import WorkGraphError


@dataclass(slots=True)
class StateSlot:
    """One versioned parameter/optimizer state instance."""

    slot_id: int
    params: dict[str, torch.Tensor]
    opt_state: OptState = field(default_factory=dict)
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    key: Hashable | None = None
    version: int = 0
    step: int = 0


def _normal_clone(value: torch.Tensor) -> torch.Tensor:
    """Return an owned copy that is NOT an inference tensor.

    Consumers execute under the engine's ``inference_mode``; tensors created
    there cannot participate in autograd later. Slot tensors must be plain
    tensors so functional updates can differentiate through them, so the copy
    runs with inference mode lifted even when the source is a normal tensor.
    """
    if torch.is_inference_mode_enabled():
        with torch.inference_mode(False):
            return value.clone()
    return value.detach().clone()


def _ensure_normal(value: torch.Tensor) -> torch.Tensor:
    """Return ``value`` if usable in autograd, else a normal-mode clone."""
    if value.is_inference():
        return _normal_clone(value)
    return value


def group_rows_by_prompt(prompt_idxs: tuple[int, ...]) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Group batch rows by prompt identity, first appearance first.

    Raises PG180 on irregular REQUEST row mapping (negative prompt idx).
    """
    groups: dict[int, list[int]] = {}
    order: list[int] = []
    for row, prompt_idx in enumerate(prompt_idxs):
        key = int(prompt_idx)
        if key < 0:
            raise WorkGraphError(
                "PG180", f"irregular REQUEST row mapping: negative prompt_idx at row {row}"
            )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    return tuple((key, tuple(groups[key])) for key in order)


def group_rows_by_request_key(
    *,
    request_ids: tuple[str, ...],
    prompt_idxs: tuple[int, ...],
    sample_idxs: tuple[int, ...],
    key_spec: PromptIdxKey | RequestStateKey,
) -> tuple[tuple[Hashable, tuple[int, ...]], ...]:
    sizes = {len(request_ids), len(prompt_idxs), len(sample_idxs)}
    if len(sizes) != 1:
        raise WorkGraphError("PG180", "irregular REQUEST row metadata lengths")
    if isinstance(key_spec, PromptIdxKey):
        return group_rows_by_prompt(prompt_idxs)
    groups: dict[Hashable, list[int]] = {}
    order: list[Hashable] = []
    for row, (request_id, prompt_idx, sample_idx) in enumerate(
        zip(request_ids, prompt_idxs, sample_idxs)
    ):
        values: list[str | int] = []
        for field_name in key_spec.fields:
            if field_name is RequestKeyField.REQUEST_ID:
                if not request_id:
                    raise WorkGraphError(
                        "PG180", f"irregular REQUEST row mapping: empty request_id at row {row}"
                    )
                values.append(str(request_id))
            elif field_name is RequestKeyField.PROMPT_IDX:
                if prompt_idx < 0:
                    raise WorkGraphError(
                        "PG180", f"irregular REQUEST row mapping: negative prompt_idx at row {row}"
                    )
                values.append(int(prompt_idx))
            else:
                if sample_idx < 0:
                    raise WorkGraphError(
                        "PG180", f"irregular REQUEST row mapping: negative sample_idx at row {row}"
                    )
                values.append(int(sample_idx))
        key: Hashable = tuple(values)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    return tuple((key, tuple(groups[key])) for key in order)


class ProgramStateBank:
    """Versioned state slots for one StateSpec."""

    def __init__(
        self,
        name: str,
        scope: StateScope,
        initial_params: dict[str, torch.Tensor],
        key_spec: PromptIdxKey | RequestStateKey | None = None,
        max_slots: int | None = None,
        max_bytes: int | None = None,
    ) -> None:
        if scope not in (StateScope.REQUEST, StateScope.REPLICA):
            raise WorkGraphError("PG100", f"unsupported state bank scope {scope!r}")
        self.name = name
        self.scope = scope
        self.key_spec = key_spec
        self._initial = {
            key: _normal_clone(value) for key, value in initial_params.items()
        }
        self._replica: StateSlot | None = None
        self._request_slots: dict[Hashable, StateSlot] = {}
        self._next_slot_id = 1
        self._max_slots = max_slots
        self._max_bytes = max_bytes

    def set_limits(
        self, *, max_slots: int | None = None, max_bytes: int | None = None
    ) -> None:
        self._max_slots = max_slots
        self._max_bytes = max_bytes

    def _fresh_slot(self, slot_id: int, key: Hashable | None = None) -> StateSlot:
        return StateSlot(
            slot_id=slot_id,
            params={key: _normal_clone(value) for key, value in self._initial.items()},
            key=key,
        )

    def slot_for(self, prompt_idx: Hashable | None) -> StateSlot:
        """Return (creating on first use) the slot for one prompt / replica."""
        if self.scope is StateScope.REPLICA:
            if self._replica is None:
                self._replica = self._fresh_slot(0)
            return self._replica
        if prompt_idx is None:
            raise WorkGraphError(
                "PG180", f"REQUEST state bank {self.name!r} requires a prompt_idx"
            )
        key: Hashable = int(prompt_idx) if isinstance(prompt_idx, int) else prompt_idx
        return self.slot_for_key(key)

    def slot_for_key(self, key: Hashable) -> StateSlot:
        if self.scope is StateScope.REPLICA:
            return self.slot_for(None)
        if key not in self._request_slots:
            if self._max_slots is not None and len(self._request_slots) >= self._max_slots:
                raise RuntimeError(
                    f"state bank {self.name!r} slot limit exceeded: "
                    f"slots={len(self._request_slots) + 1} max_slots={self._max_slots}"
                )
            self._request_slots[key] = self._fresh_slot(self._next_slot_id, key)
            self._next_slot_id += 1
        return self._request_slots[key]

    def get_slot(self, key: Hashable) -> StateSlot | None:
        if self.scope is StateScope.REPLICA:
            return self._replica
        return self._request_slots.get(key)

    def keys(self) -> tuple[Hashable, ...]:
        return tuple(self._request_slots)

    def key_for_slot_id(self, slot_id: int) -> Hashable | None:
        """Resolve an internal numeric slot id back to its public state key."""

        if self.scope is StateScope.REPLICA:
            if self._replica is None or self._replica.slot_id != int(slot_id):
                raise KeyError(slot_id)
            return None
        for key, slot in self._request_slots.items():
            if slot.slot_id == int(slot_id):
                return key
        raise KeyError(slot_id)

    def clear(self, key: Hashable) -> None:
        self._request_slots.pop(key, None)

    def groups_for_metadata(
        self,
        *,
        request_ids: tuple[str, ...],
        prompt_idxs: tuple[int, ...],
        sample_idxs: tuple[int, ...],
    ) -> tuple[tuple[Hashable, tuple[int, ...]], ...]:
        if self.scope is StateScope.REPLICA:
            return ((0, tuple(range(len(request_ids)))),)
        key_spec = self.key_spec
        if not isinstance(key_spec, (PromptIdxKey, RequestStateKey)):
            raise WorkGraphError("PG180", f"REQUEST state bank {self.name!r} has no key spec")
        return group_rows_by_request_key(
            request_ids=request_ids,
            prompt_idxs=prompt_idxs,
            sample_idxs=sample_idxs,
            key_spec=key_spec,
        )

    @staticmethod
    def _tensor_bytes(value: torch.Tensor) -> int:
        return int(value.numel() * value.element_size())

    @property
    def estimated_bytes(self) -> int:
        slots = ([] if self._replica is None else [self._replica]) + list(
            self._request_slots.values()
        )
        return sum(
            self._tensor_bytes(value)
            for slot in slots
            for value in (*slot.params.values(), *slot.buffers.values())
        )

    def publish_buffers(self, slot: StateSlot, buffers: dict[str, torch.Tensor]) -> int:
        owned = {name: _normal_clone(value) for name, value in buffers.items()}
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

    def publish_update(
        self,
        slot: StateSlot,
        params: dict[str, torch.Tensor],
        opt_state: OptState,
    ) -> int:
        """Install one functional update; returns the new version."""
        slot.params = {key: _ensure_normal(value) for key, value in params.items()}
        slot.opt_state = {
            key: {sk: _ensure_normal(sv) for sk, sv in state.items()}
            for key, state in opt_state.items()
        }
        slot.version += 1
        slot.step += 1
        return slot.version


__all__ = [
    "ProgramStateBank",
    "StateSlot",
    "group_rows_by_prompt",
    "group_rows_by_request_key",
]
