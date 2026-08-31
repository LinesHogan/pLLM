"""Closed contracts for generational request-scoped state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch

from tllm.contracts.request_identity import (
    RequestIdentity,
    RequestKeyField,
    canonical_request_key_fields,
)
from tllm.contracts.state_ref import LineageId, VersionId


@dataclass(frozen=True, slots=True)
class RequestStateKey:
    """Canonical projection of one identity under a signed key declaration."""

    fields: tuple[RequestKeyField, ...]
    request_id: str
    prompt_idx: int
    sample_idx: int

    @classmethod
    def project(
        cls,
        identity: RequestIdentity,
        fields: tuple[RequestKeyField, ...],
    ) -> "RequestStateKey":
        fields = canonical_request_key_fields(fields)
        if RequestKeyField.REQUEST_ID in fields and not identity.request_id:
            raise ValueError("request_id state key requires a non-empty request id")
        if RequestKeyField.PROMPT_IDX in fields and identity.prompt_idx < 0:
            raise ValueError("prompt_idx state key requires resolved prompt metadata")
        if RequestKeyField.SAMPLE_IDX in fields and identity.sample_idx < 0:
            raise ValueError("sample_idx state key requires resolved sample metadata")
        return cls(
            fields=fields,
            request_id=(
                identity.request_id
                if RequestKeyField.REQUEST_ID in fields
                else ""
            ),
            prompt_idx=(
                identity.prompt_idx
                if RequestKeyField.PROMPT_IDX in fields
                else -1
            ),
            sample_idx=(
                identity.sample_idx
                if RequestKeyField.SAMPLE_IDX in fields
                else -1
            ),
        )


@dataclass(frozen=True, slots=True, order=True)
class StateSlotRef:
    slot_id: int
    generation: int

    def __post_init__(self) -> None:
        if self.slot_id < 0:
            raise ValueError("request state slot id must be non-negative")
        if self.generation < 1:
            raise ValueError("request state slot generation must be positive")


@dataclass(frozen=True, slots=True)
class RequestStateLease:
    key: RequestStateKey
    slot: StateSlotRef
    lineage_id: LineageId
    epoch: int


@dataclass(frozen=True, slots=True)
class RequestStateSelectionRow:
    identity: RequestIdentity
    lease: RequestStateLease
    canonical_row_index: int

    def __post_init__(self) -> None:
        if self.canonical_row_index < 0:
            raise ValueError(
                "request-state selection row index must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class RequestStateSelectionBatch:
    resource: str
    selection_id: int
    epoch: int
    rows: tuple[RequestStateSelectionRow, ...]
    leases: tuple[RequestStateLease, ...]
    before_versions: tuple[VersionId, ...]
    device_slot_ids: torch.Tensor
    device_lease_slot_ids: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class RequestStateLossBatch:
    """One closed optimizer payload aligned to a request-state selection."""

    selection: RequestStateSelectionBatch
    losses: tuple[torch.Tensor, ...]

    def __post_init__(self) -> None:
        if len(self.losses) != len(self.selection.leases):
            raise ValueError(
                "request-state loss batch must align with unique selection leases"
            )
        if any(loss.numel() != 1 for loss in self.losses):
            raise ValueError("request-state loss batch entries must be scalar")


@dataclass(frozen=True, slots=True)
class RequestIdentityBatch:
    """One exact backend-resolved identity batch for a native step."""

    identities: tuple[RequestIdentity, ...]

    def __post_init__(self) -> None:
        if any(
            not identity.request_id
            or identity.prompt_idx < 0
            or identity.sample_idx < 0
            for identity in self.identities
        ):
            raise ValueError(
                "request identity batch requires fully resolved identities"
            )


@dataclass(frozen=True, slots=True)
class IndexedRequestIdentityRow:
    """One identity projected from its canonical source tensor row."""

    canonical_row_index: int
    identity: RequestIdentity

    def __post_init__(self) -> None:
        if self.canonical_row_index < 0:
            raise ValueError(
                "indexed request identity row must be non-negative"
            )
        if (
            not self.identity.request_id
            or self.identity.prompt_idx < 0
            or self.identity.sample_idx < 0
        ):
            raise ValueError(
                "indexed request identity requires a fully resolved identity"
            )


@dataclass(frozen=True, slots=True)
class IndexedRequestIdentityBatch:
    """A phase projection retaining exact rows in a larger tensor batch."""

    rows: tuple[IndexedRequestIdentityRow, ...]

    def __post_init__(self) -> None:
        indices = tuple(row.canonical_row_index for row in self.rows)
        if indices != tuple(sorted(set(indices))):
            raise ValueError(
                "indexed request identity rows must be strictly increasing"
            )

    @property
    def identities(self) -> tuple[RequestIdentity, ...]:
        return tuple(row.identity for row in self.rows)


class StateEffectOperation(StrEnum):
    ADMIT = "admit"
    READ = "read"
    TRANSITION = "transition"
    RETIRE = "retire"


@dataclass(frozen=True, slots=True)
class StateSlotEffect:
    key: RequestStateKey
    slot: StateSlotRef
    lineage_id: LineageId
    before_version: VersionId
    after_version: VersionId


@dataclass(frozen=True, slots=True)
class StateEffectReceipt:
    receipt_id: int
    resource: str
    operation: StateEffectOperation
    selection_id: int
    epoch: int
    slots: tuple[StateSlotEffect, ...]


@dataclass(frozen=True, slots=True)
class RequestStateVersion:
    key: RequestStateKey
    slot: StateSlotRef
    lineage_id: LineageId
    version: VersionId


@dataclass(frozen=True, slots=True)
class RequestStateEffectAudit:
    """Typed before-image for one signed request-state Region effect."""

    resource: str
    epoch: int
    step_id: int
    selection: RequestStateSelectionBatch
    operation: StateEffectOperation
    selected_before: tuple[StateSlotEffect, ...]
    store_before: tuple[RequestStateVersion, ...]


@dataclass(frozen=True, slots=True)
class RequestStateAcquisition:
    selection: RequestStateSelectionBatch
    admission: StateEffectReceipt


@dataclass(frozen=True, slots=True)
class RequestStateAsyncPin:
    resource: str
    pin_id: int
    epoch: int
    versions: tuple[RequestStateVersion, ...]


__all__ = [
    "IndexedRequestIdentityBatch",
    "IndexedRequestIdentityRow",
    "RequestIdentityBatch",
    "RequestStateAcquisition",
    "RequestStateAsyncPin",
    "RequestStateEffectAudit",
    "RequestStateKey",
    "RequestStateLease",
    "RequestStateLossBatch",
    "RequestStateSelectionBatch",
    "RequestStateSelectionRow",
    "RequestStateVersion",
    "StateEffectOperation",
    "StateEffectReceipt",
    "StateSlotEffect",
    "StateSlotRef",
]
