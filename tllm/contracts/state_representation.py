"""Closed physical provider plans for versioned tensor state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json

from tllm.contracts.state_ref import LayoutId


class StateRepresentation(StrEnum):
    RESIDENT_MODULE = "resident_module"
    DENSE_PAGE_ARENA = "dense_page_arena"
    SPARSE_PAGE_ARENA = "sparse_page_arena"
    VERSIONED_OVERLAY = "versioned_overlay"
    SHARDED_STATE = "sharded_state"
    HOST_SERVICE = "host_service"
    EXTERNAL_HANDLE = "external_handle"


@dataclass(frozen=True, slots=True)
class StateProviderPlan:
    """One physical candidate, or the identical Rust-selected plan."""

    resource: str
    provider_id: str
    representation: StateRepresentation
    layout_id: LayoutId
    layout_namespace: str
    layout_digest: str
    max_admitted_lineages: int
    max_live_versions_per_lineage: int
    root_page_count: int
    reserved_bytes: int
    capture_stable: bool
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.resource or not self.provider_id:
            raise ValueError("state provider plan requires resource and provider IDs")
        if int(self.layout_id) < 0:
            raise ValueError("state provider layout ID must be non-negative")
        for value in (self.layout_namespace, self.layout_digest):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    "state provider layout namespace/digest must be lowercase SHA-256"
                )
        if (
            self.max_admitted_lineages < 1
            or self.max_live_versions_per_lineage < 1
            or self.root_page_count < 1
            or self.reserved_bytes < 0
        ):
            raise ValueError("state provider capacity facts are invalid")
        if (
            tuple(sorted(self.capabilities)) != self.capabilities
            or len(set(self.capabilities)) != len(self.capabilities)
            or any(not capability for capability in self.capabilities)
        ):
            raise ValueError(
                "state provider capabilities must be sorted unique names"
            )

    @property
    def candidate_key(self) -> tuple[str, str]:
        return (self.resource, self.provider_id)


@dataclass(frozen=True, slots=True)
class StateLayoutRequirement:
    """Representation-neutral frozen facts for one request state."""

    resource: str
    layout_id: LayoutId
    layout_namespace: str
    layout_digest: str
    max_admitted_lineages: int
    required_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.resource or int(self.layout_id) < 0:
            raise ValueError("state layout requirement has an invalid identity")
        for value in (self.layout_namespace, self.layout_digest):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    "state layout namespace/digest must be lowercase SHA-256"
                )
        if self.max_admitted_lineages < 1:
            raise ValueError("state layout requirement needs admission capacity")
        if (
            tuple(sorted(self.required_capabilities))
            != self.required_capabilities
            or len(set(self.required_capabilities))
            != len(self.required_capabilities)
            or any(not capability for capability in self.required_capabilities)
        ):
            raise ValueError(
                "state layout requirements must be sorted unique capabilities"
            )


@dataclass(frozen=True, slots=True)
class StateRepresentationRequest:
    memory_budget_bytes: int
    states: tuple[StateLayoutRequirement, ...]
    candidates: tuple[StateProviderPlan, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.memory_budget_bytes < 0 or self.schema_version != 1:
            raise ValueError("invalid state representation request envelope")
        resources = tuple(state.resource for state in self.states)
        candidate_keys = tuple(candidate.candidate_key for candidate in self.candidates)
        if tuple(sorted(resources)) != resources or len(set(resources)) != len(resources):
            raise ValueError(
                "state representation requirements need sorted unique resources"
            )
        if (
            tuple(sorted(candidate_keys)) != candidate_keys
            or len(set(candidate_keys)) != len(candidate_keys)
        ):
            raise ValueError(
                "state provider candidates need sorted unique resource/provider keys"
            )


@dataclass(frozen=True, slots=True)
class StateRepresentationDecision:
    request_digest: str
    plans: tuple[StateProviderPlan, ...]
    reserved_bytes: int
    signature: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for value, label in (
            (self.request_digest, "request digest"),
            (self.signature, "signature"),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"state representation {label} must be SHA-256")
        resources = tuple(plan.resource for plan in self.plans)
        if tuple(sorted(resources)) != resources or len(set(resources)) != len(resources):
            raise ValueError(
                "state representation decision needs sorted unique resources"
            )
        if self.reserved_bytes < 0 or self.schema_version != 1:
            raise ValueError("invalid state representation decision envelope")


@dataclass(frozen=True, slots=True)
class StatePhysicalPlan:
    plan_epoch: int
    request_digest: str
    representation_signature: str
    representations: tuple[StateProviderPlan, ...]
    reserved_bytes: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.plan_epoch < 1 or self.reserved_bytes < 0 or self.schema_version != 1:
            raise ValueError("invalid physical state plan envelope")
        for value in (self.request_digest, self.representation_signature):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    "physical state plan request/signature must be SHA-256"
                )
        resources = tuple(plan.resource for plan in self.representations)
        if tuple(sorted(resources)) != resources or len(set(resources)) != len(resources):
            raise ValueError(
                "physical state plan needs sorted unique representations"
            )


def encode_state_provider_plans(plans: tuple[StateProviderPlan, ...]) -> str:
    """Encode the closed local-candidate wire consumed by Rust."""

    return json.dumps(
        [
            {
                "resource": plan.resource,
                "provider_id": plan.provider_id,
                "representation": plan.representation.value,
                "layout_id": int(plan.layout_id),
                "layout_namespace": plan.layout_namespace,
                "layout_digest": plan.layout_digest,
                "max_admitted_lineages": plan.max_admitted_lineages,
                "max_live_versions_per_lineage": (
                    plan.max_live_versions_per_lineage
                ),
                "root_page_count": plan.root_page_count,
                "reserved_bytes": plan.reserved_bytes,
                "capture_stable": plan.capture_stable,
                "capabilities": list(plan.capabilities),
            }
            for plan in plans
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


__all__ = [
    "StateLayoutRequirement",
    "StatePhysicalPlan",
    "StateProviderPlan",
    "StateRepresentation",
    "StateRepresentationDecision",
    "StateRepresentationRequest",
    "encode_state_provider_plans",
]
