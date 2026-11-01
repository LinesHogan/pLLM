"""Closed contracts for generational request-scoped state."""

from __future__ import annotations

import functools
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
    prompt_index: int
    sample_index: int
    route: str = ""

    @classmethod
    @functools.lru_cache(maxsize=1 << 16)  # identities repeat every decode step
    def project(
        cls,
        identity: RequestIdentity,
        fields: tuple[RequestKeyField, ...],
    ) -> "RequestStateKey":
        fields = canonical_request_key_fields(fields)
        if (
            RequestKeyField.REQUEST_ID in fields
            and not identity.request_id
            or RequestKeyField.PROMPT_INDEX in fields
            and identity.prompt_index < 0
            or RequestKeyField.SAMPLE_INDEX in fields
            and identity.sample_index < 0
            or RequestKeyField.ROUTE in fields
            and not identity.route
        ):
            raise ValueError(f"RequestIdentity {identity} is missing a field required by {fields}")
        return cls(
            fields=fields,
            request_id=(identity.request_id if RequestKeyField.REQUEST_ID in fields else ""),
            prompt_index=(identity.prompt_index if RequestKeyField.PROMPT_INDEX in fields else -1),
            sample_index=(identity.sample_index if RequestKeyField.SAMPLE_INDEX in fields else -1),
            route=(identity.route if RequestKeyField.ROUTE in fields else ""),
        )
