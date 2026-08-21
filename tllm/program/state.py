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

import torch

from tllm.program.optim import OptState
from tllm.workgraph.frontend import StateScope
from tllm.workgraph.lowering import WorkGraphError


@dataclass(slots=True)
class StateSlot:
    """One versioned parameter/optimizer state instance."""

    slot_id: int
    params: dict[str, torch.Tensor]
    opt_state: OptState = field(default_factory=dict)
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


class ProgramStateBank:
    """Versioned state slots for one StateSpec."""

    def __init__(
        self,
        name: str,
        scope: StateScope,
        initial_params: dict[str, torch.Tensor],
    ) -> None:
        if scope not in (StateScope.REQUEST, StateScope.REPLICA):
            raise WorkGraphError("PG100", f"unsupported state bank scope {scope!r}")
        self.name = name
        self.scope = scope
        self._initial = {
            key: _normal_clone(value) for key, value in initial_params.items()
        }
        self._replica: StateSlot | None = None
        self._request_slots: dict[int, StateSlot] = {}

    def _fresh_slot(self, slot_id: int) -> StateSlot:
        return StateSlot(
            slot_id=slot_id,
            params={key: _normal_clone(value) for key, value in self._initial.items()},
        )

    def slot_for(self, prompt_idx: int | None) -> StateSlot:
        """Return (creating on first use) the slot for one prompt / replica."""
        if self.scope is StateScope.REPLICA:
            if self._replica is None:
                self._replica = self._fresh_slot(0)
            return self._replica
        if prompt_idx is None:
            raise WorkGraphError(
                "PG180", f"REQUEST state bank {self.name!r} requires a prompt_idx"
            )
        key = int(prompt_idx)
        if key not in self._request_slots:
            self._request_slots[key] = self._fresh_slot(key)
        return self._request_slots[key]

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

    @property
    def slot_count(self) -> int:
        if self.scope is StateScope.REPLICA:
            return 1 if self._replica is not None else 0
        return len(self._request_slots)


__all__ = ["ProgramStateBank", "StateSlot", "group_rows_by_prompt"]
