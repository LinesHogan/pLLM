"""Closed contracts for generational request-scoped state."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.contracts.request_identity import (
    RequestIdentity,
    RequestKeyField,
    canonical_request_key_fields,
)


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
