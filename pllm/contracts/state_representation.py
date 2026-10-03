"""Closed physical provider plans for versioned tensor state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pllm.contracts.state_ref import LayoutId


class StateRepresentation(StrEnum):
    RESIDENT_MODULE = "resident_module"
    DENSE_PAGE_ARENA = "dense_page_arena"
    SPARSE_PAGE_ARENA = "sparse_page_arena"
    VERSIONED_OVERLAY = "versioned_overlay"
    SHARDED_STATE = "sharded_state"
    HOST_SERVICE = "host_service"
    EXTERNAL_HANDLE = "external_handle"


class StateDeviceKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True, slots=True)
class StateProviderPlan:
    """One physical candidate, or the identical Rust-selected plan."""

    resource: str
    provider_id: str
    representation: StateRepresentation
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
    request_digest: str
    plans: tuple[StateProviderPlan, ...]
    reserved_bytes: int
    signature: str
    schema_version: int = 1


@dataclass(frozen=True, slots=True)
class StatePhysicalPlan:
    plan_epoch: int
    request_digest: str
    representation_signature: str
    representations: tuple[StateProviderPlan, ...]
    reserved_bytes: int
    schema_version: int = 1


__all__ = [
    "StateDeviceKind",
    "StateLayoutRequirement",
    "StatePhysicalPlan",
    "StateProviderPlan",
    "StateRepresentation",
    "StateRepresentationDecision",
    "StateRepresentationRequest",
]
