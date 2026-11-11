"""Closed physical provider plans for versioned tensor state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pllm.contracts.state_ref import LayoutId


class StateDeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class StateProviderPlan:
    """One physical candidate, or the identical Rust-selected plan."""

    resource: str
    provider_id: str
    device_kind: StateDeviceKind
    layout_id: LayoutId
    max_admitted_lineages: int
    # Pages per field, in layout order: older versions keep only the fields their readers pin.
    page_capacities: tuple[int, ...]
    reserved_bytes: int
    capture_stable: bool
    capabilities: tuple[str, ...] = ()
    host_bytes: int = 0


@dataclass(frozen=True, slots=True)
class StateRepresentationRequest:
    """Choose one provider candidate per state resource, in `resources` order, within the page budget."""

    state_page_budget_bytes: int
    resources: tuple[str, ...]
    candidates: tuple[StateProviderPlan, ...]


@dataclass(frozen=True, slots=True)
class StateRepresentationDecision:
    plans: tuple[StateProviderPlan, ...]
    reserved_bytes: int


__all__ = [
    "StateDeviceKind",
    "StateProviderPlan",
    "StateRepresentationDecision",
    "StateRepresentationRequest",
]
