"""Closed contracts for generational request-scoped state."""

from __future__ import annotations

from dataclasses import dataclass

from pllm.contracts.request_identity import (
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
    route: str = ""

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
        if RequestKeyField.ROUTE in fields and not identity.route:
            raise ValueError("route state key requires a user-computed route value")
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
            route=(
                identity.route
                if RequestKeyField.ROUTE in fields
                else ""
            ),
        )
