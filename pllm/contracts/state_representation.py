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
    device_ordinal: int
    layout_id: LayoutId
    layout_namespace: str
    layout_digest: str
    max_admitted_lineages: int
    page_capacity: int
    root_page_count: int
    reserved_bytes: int
    capture_stable: bool
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StateLayoutRequirement:
    """Representation-neutral frozen facts for one request state."""

    resource: str
    layout_id: LayoutId
    layout_namespace: str
    layout_digest: str
    max_admitted_lineages: int
    required_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StateRepresentationRequest:
    memory_budget_bytes: int
    states: tuple[StateLayoutRequirement, ...]
    candidates: tuple[StateProviderPlan, ...]
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class StateRepresentationDecision:
    plans: tuple[StateProviderPlan, ...]
    reserved_bytes: int


__all__ = [
    "StateDeviceKind",
    "StateLayoutRequirement",
    "StateProviderPlan",
    "StateRepresentationDecision",
    "StateRepresentationRequest",
]
